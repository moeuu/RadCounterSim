#include <embree4/rtcore.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

namespace {

thread_local std::string last_embree_error;

void embree_error(void *, RTCError code, const char *message) noexcept {
  if (code == RTC_ERROR_NONE) {
    return;
  }
  last_embree_error = message == nullptr ? "unknown" : message;
}

void throw_device_error(RTCDevice device, const char *operation) {
  const RTCError code = rtcGetDeviceError(device);
  if (code == RTC_ERROR_NONE) {
    return;
  }
  throw std::runtime_error(
      std::string(operation) + " failed with Embree error " +
      std::to_string(static_cast<int>(code)) + ": " + last_embree_error);
}

void require_shape(const py::buffer_info &info, py::ssize_t columns,
                   const char *name) {
  if (info.ndim != 2 || info.shape[1] != columns) {
    throw std::invalid_argument(std::string(name) + " must have shape (N, " +
                                std::to_string(columns) + ")");
  }
}

class EmbreeScene {
public:
  EmbreeScene() : device_(rtcNewDevice(nullptr)), scene_(nullptr) {
    if (device_ == nullptr) {
      throw std::runtime_error("failed to create Embree device");
    }
    rtcSetDeviceErrorFunction(device_, embree_error, nullptr);
    scene_ = rtcNewScene(device_);
    if (scene_ == nullptr) {
      rtcReleaseDevice(device_);
      device_ = nullptr;
      throw std::runtime_error("failed to create Embree scene");
    }
    rtcSetSceneBuildQuality(scene_, RTC_BUILD_QUALITY_MEDIUM);
  }

  ~EmbreeScene() {
    if (scene_ != nullptr) {
      rtcReleaseScene(scene_);
    }
    if (device_ != nullptr) {
      rtcReleaseDevice(device_);
    }
  }

  EmbreeScene(const EmbreeScene &) = delete;
  EmbreeScene &operator=(const EmbreeScene &) = delete;

  std::uint32_t
  add_triangle_mesh(
      const py::array_t<float, py::array::c_style | py::array::forcecast>
          &vertices_m,
      const py::array_t<std::uint32_t,
                        py::array::c_style | py::array::forcecast>
          &triangles,
      std::uint32_t material_index) {
    const auto vertex_info = vertices_m.request();
    const auto triangle_info = triangles.request();
    require_shape(vertex_info, 3, "vertices_m");
    require_shape(triangle_info, 3, "triangles");
    if (vertex_info.shape[0] == 0 || triangle_info.shape[0] == 0) {
      throw std::invalid_argument("triangle meshes must not be empty");
    }

    RTCGeometry geometry = rtcNewGeometry(device_, RTC_GEOMETRY_TYPE_TRIANGLE);
    if (geometry == nullptr) {
      throw std::runtime_error("failed to create Embree geometry");
    }
    auto *vertex_buffer = static_cast<float *>(rtcSetNewGeometryBuffer(
        geometry, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3,
        3 * sizeof(float), static_cast<std::size_t>(vertex_info.shape[0])));
    auto *index_buffer = static_cast<std::uint32_t *>(rtcSetNewGeometryBuffer(
        geometry, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3,
        3 * sizeof(std::uint32_t),
        static_cast<std::size_t>(triangle_info.shape[0])));
    if (vertex_buffer == nullptr || index_buffer == nullptr) {
      rtcReleaseGeometry(geometry);
      throw std::runtime_error("failed to allocate Embree geometry buffers");
    }
    std::memcpy(vertex_buffer, vertex_info.ptr,
                static_cast<std::size_t>(vertex_info.size) * sizeof(float));
    std::memcpy(index_buffer, triangle_info.ptr,
                static_cast<std::size_t>(triangle_info.size) *
                    sizeof(std::uint32_t));
    rtcCommitGeometry(geometry);
    throw_device_error(device_, "rtcCommitGeometry");
    const std::uint32_t geometry_id = rtcAttachGeometry(scene_, geometry);
    rtcReleaseGeometry(geometry);
    if (geometry_id == RTC_INVALID_GEOMETRY_ID) {
      throw std::runtime_error("failed to attach Embree geometry");
    }
    if (geometry_material_.size() <= geometry_id) {
      geometry_material_.resize(static_cast<std::size_t>(geometry_id) + 1, 0);
    }
    geometry_material_[geometry_id] = material_index;
    committed_ = false;
    return geometry_id;
  }

  void commit() {
    rtcCommitScene(scene_);
    throw_device_error(device_, "rtcCommitScene");
    committed_ = true;
    ++revision_;
  }

  [[nodiscard]] std::uint64_t revision() const { return revision_; }

  py::array_t<double> trace_transmission(
      const py::array_t<double, py::array::c_style | py::array::forcecast>
          &origins_m,
      const py::array_t<double, py::array::c_style | py::array::forcecast>
          &targets_m,
      const py::array_t<double, py::array::c_style | py::array::forcecast>
          &attenuation_per_m) const {
    if (!committed_) {
      throw std::logic_error("commit() must be called before tracing");
    }
    const auto origin_info = origins_m.request();
    const auto target_info = targets_m.request();
    const auto attenuation_info = attenuation_per_m.request();
    require_shape(origin_info, 3, "origins_m");
    require_shape(target_info, 3, "targets_m");
    if (origin_info.shape[0] != target_info.shape[0]) {
      throw std::invalid_argument("origins_m and targets_m must have equal rows");
    }
    if (attenuation_info.ndim != 2 || attenuation_info.shape[1] == 0) {
      throw std::invalid_argument(
          "attenuation_per_m must have shape (material, energy_bin)");
    }
    for (std::uint32_t material : geometry_material_) {
      if (material >= static_cast<std::uint32_t>(attenuation_info.shape[0])) {
        throw std::invalid_argument(
            "attenuation_per_m does not contain every mesh material index");
      }
    }

    const auto ray_count = static_cast<std::size_t>(origin_info.shape[0]);
    const auto material_count =
        static_cast<std::size_t>(attenuation_info.shape[0]);
    const auto energy_count =
        static_cast<std::size_t>(attenuation_info.shape[1]);
    py::array_t<double> result(
        {static_cast<py::ssize_t>(ray_count),
         static_cast<py::ssize_t>(energy_count)});
    auto result_info = result.request();
    const auto *origin = static_cast<const double *>(origin_info.ptr);
    const auto *target = static_cast<const double *>(target_info.ptr);
    const auto *attenuation =
        static_cast<const double *>(attenuation_info.ptr);
    auto *output = static_cast<double *>(result_info.ptr);

    py::gil_scoped_release release;
    for (std::size_t ray_index = 0; ray_index < ray_count; ++ray_index) {
      const double delta_x = target[3 * ray_index] - origin[3 * ray_index];
      const double delta_y = target[3 * ray_index + 1] - origin[3 * ray_index + 1];
      const double delta_z = target[3 * ray_index + 2] - origin[3 * ray_index + 2];
      const double length_m =
          std::sqrt(delta_x * delta_x + delta_y * delta_y + delta_z * delta_z);
      std::vector<double> material_path_m(material_count, 0.0);
      if (length_m > 0.0) {
        const float direction_x = static_cast<float>(delta_x / length_m);
        const float direction_y = static_cast<float>(delta_y / length_m);
        const float direction_z = static_cast<float>(delta_z / length_m);
        float next_tnear = 0.0F;
        std::unordered_map<std::uint32_t, float> entry_distance;
        for (std::size_t hit_count = 0; hit_count < 4096; ++hit_count) {
          RTCRayHit ray_hit{};
          ray_hit.ray.org_x = static_cast<float>(origin[3 * ray_index]);
          ray_hit.ray.org_y = static_cast<float>(origin[3 * ray_index + 1]);
          ray_hit.ray.org_z = static_cast<float>(origin[3 * ray_index + 2]);
          ray_hit.ray.dir_x = direction_x;
          ray_hit.ray.dir_y = direction_y;
          ray_hit.ray.dir_z = direction_z;
          ray_hit.ray.tnear = next_tnear;
          ray_hit.ray.tfar = static_cast<float>(length_m);
          ray_hit.ray.mask = 0xFFFFFFFFU;
          ray_hit.ray.flags = 0;
          ray_hit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
          ray_hit.hit.primID = RTC_INVALID_GEOMETRY_ID;
          RTCIntersectArguments arguments;
          rtcInitIntersectArguments(&arguments);
          rtcIntersect1(scene_, &ray_hit, &arguments);
          if (ray_hit.hit.geomID == RTC_INVALID_GEOMETRY_ID) {
            break;
          }
          const std::uint32_t geometry_id = ray_hit.hit.geomID;
          const float hit_distance = ray_hit.ray.tfar;
          const auto active = entry_distance.find(geometry_id);
          if (active == entry_distance.end()) {
            entry_distance.emplace(geometry_id, hit_distance);
          } else {
            const std::uint32_t material = geometry_material_.at(geometry_id);
            material_path_m[material] +=
                std::max(0.0, static_cast<double>(hit_distance - active->second));
            entry_distance.erase(active);
          }
          const float epsilon =
              std::max(1.0e-5F, 1.0e-5F * std::abs(hit_distance));
          next_tnear = hit_distance + epsilon;
          if (next_tnear >= static_cast<float>(length_m)) {
            break;
          }
        }
        for (const auto &[geometry_id, entry] : entry_distance) {
          const std::uint32_t material = geometry_material_.at(geometry_id);
          material_path_m[material] +=
              std::max(0.0, length_m - static_cast<double>(entry));
        }
      }
      for (std::size_t energy = 0; energy < energy_count; ++energy) {
        double optical_depth = 0.0;
        for (std::size_t material = 0; material < material_count; ++material) {
          optical_depth += material_path_m[material] *
                           attenuation[material * energy_count + energy];
        }
        output[ray_index * energy_count + energy] = std::exp(-optical_depth);
      }
    }
    throw_device_error(device_, "rtcIntersect1");
    return result;
  }

private:
  RTCDevice device_;
  RTCScene scene_;
  std::vector<std::uint32_t> geometry_material_;
  bool committed_ = false;
  std::uint64_t revision_ = 0;
};

} // namespace

PYBIND11_MODULE(_radcounter_embree, module) {
  module.doc() = "Embree 4 segment attenuation backend for RadCounterSim";
  module.def("embree_version", []() { return RTC_VERSION_STRING; });
  py::class_<EmbreeScene>(module, "Scene")
      .def(py::init<>())
      .def("add_triangle_mesh", &EmbreeScene::add_triangle_mesh,
           py::arg("vertices_m"), py::arg("triangles"),
           py::arg("material_index"))
      .def("commit", &EmbreeScene::commit)
      .def_property_readonly("revision", &EmbreeScene::revision)
      .def("trace_transmission", &EmbreeScene::trace_transmission,
           py::arg("origins_m"), py::arg("targets_m"),
           py::arg("attenuation_per_m"));
}
