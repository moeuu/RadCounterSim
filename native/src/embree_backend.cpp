#include <embree4/rtcore.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

thread_local std::string last_embree_error;

void embree_error(void *, RTCError code, const char *message) noexcept {
  if (code != RTC_ERROR_NONE) {
    last_embree_error = message == nullptr ? "unknown" : message;
  }
}

void throw_device_error(RTCDevice device, const char *operation) {
  const RTCError code = rtcGetDeviceError(device);
  if (code != RTC_ERROR_NONE) {
    throw std::runtime_error(
        std::string(operation) + " failed with Embree error " +
        std::to_string(static_cast<int>(code)) + ": " + last_embree_error);
  }
}

void require_shape(const py::buffer_info &info, py::ssize_t columns,
                   const char *name) {
  if (info.ndim != 2 || info.shape[1] != columns) {
    throw std::invalid_argument(std::string(name) + " must have shape (N, " +
                                std::to_string(columns) + ")");
  }
}

enum class GeometryMode { Solid, ThinSheet };

struct GeometryRecord {
  std::uint32_t material_index = 0;
  GeometryMode mode = GeometryMode::Solid;
  double thickness_m = 0.0;
  double grazing_cosine_floor = 0.05;
  RTCScene object_scene = nullptr;
};

template <typename Function>
void parallel_for(std::size_t count, Function &&function) {
  const unsigned int available = std::max(1U, std::thread::hardware_concurrency());
  const std::size_t worker_count =
      count < 64 ? 1 : std::min<std::size_t>(available, count);
  if (worker_count == 1) {
    for (std::size_t index = 0; index < count; ++index) {
      function(index);
    }
    return;
  }
  std::atomic<std::size_t> next{0};
  std::vector<std::thread> workers;
  workers.reserve(worker_count);
  for (std::size_t worker = 0; worker < worker_count; ++worker) {
    workers.emplace_back([&]() {
      for (;;) {
        const std::size_t index = next.fetch_add(1, std::memory_order_relaxed);
        if (index >= count) {
          return;
        }
        function(index);
      }
    });
  }
  for (auto &worker : workers) {
    worker.join();
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
    rtcSetSceneFlags(scene_, RTC_SCENE_FLAG_DYNAMIC);
  }

  ~EmbreeScene() {
    if (scene_ != nullptr) {
      rtcReleaseScene(scene_);
    }
    for (const auto &[geometry_id, record] : geometry_records_) {
      static_cast<void>(geometry_id);
      if (record.object_scene != nullptr) {
        rtcReleaseScene(record.object_scene);
      }
    }
    if (device_ != nullptr) {
      rtcReleaseDevice(device_);
    }
  }

  EmbreeScene(const EmbreeScene &) = delete;
  EmbreeScene &operator=(const EmbreeScene &) = delete;

  std::uint32_t add_triangle_mesh(
      const py::array_t<float, py::array::c_style | py::array::forcecast>
          &vertices_m,
      const py::array_t<std::uint32_t,
                        py::array::c_style | py::array::forcecast> &triangles,
      std::uint32_t material_index,
      const std::string &geometry_mode = "solid", double thickness_m = 0.0,
      double grazing_cosine_floor = 0.05) {
    const auto vertex_info = vertices_m.request();
    const auto triangle_info = triangles.request();
    require_shape(vertex_info, 3, "vertices_m");
    require_shape(triangle_info, 3, "triangles");
    if (vertex_info.shape[0] == 0 || triangle_info.shape[0] == 0) {
      throw std::invalid_argument("triangle meshes must not be empty");
    }
    GeometryMode mode;
    if (geometry_mode == "solid") {
      mode = GeometryMode::Solid;
    } else if (geometry_mode == "thin_sheet") {
      mode = GeometryMode::ThinSheet;
      if (!(thickness_m > 0.0) || !std::isfinite(thickness_m)) {
        throw std::invalid_argument(
            "thin-sheet geometry requires positive finite thickness_m");
      }
      if (!(grazing_cosine_floor > 0.0 && grazing_cosine_floor <= 1.0)) {
        throw std::invalid_argument(
            "grazing_cosine_floor must be in the interval (0, 1]");
      }
    } else {
      throw std::invalid_argument("geometry_mode must be solid or thin_sheet");
    }
    const auto *input_indices =
        static_cast<const std::uint32_t *>(triangle_info.ptr);
    for (py::ssize_t index = 0; index < triangle_info.size; ++index) {
      if (input_indices[index] >=
          static_cast<std::uint32_t>(vertex_info.shape[0])) {
        throw std::invalid_argument("triangle index is outside vertices_m");
      }
    }

    std::unique_lock lock(mutex_);
    RTCScene object_scene = rtcNewScene(device_);
    if (object_scene == nullptr) {
      throw std::runtime_error("failed to create Embree object scene");
    }
    rtcSetSceneBuildQuality(object_scene, RTC_BUILD_QUALITY_MEDIUM);
    RTCGeometry geometry = rtcNewGeometry(device_, RTC_GEOMETRY_TYPE_TRIANGLE);
    if (geometry == nullptr) {
      rtcReleaseScene(object_scene);
      throw std::runtime_error("failed to create Embree geometry");
    }
    rtcSetGeometryBuildQuality(geometry, RTC_BUILD_QUALITY_REFIT);
    auto *vertex_buffer = static_cast<float *>(rtcSetNewGeometryBuffer(
        geometry, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3,
        3 * sizeof(float), static_cast<std::size_t>(vertex_info.shape[0])));
    auto *index_buffer = static_cast<std::uint32_t *>(rtcSetNewGeometryBuffer(
        geometry, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3,
        3 * sizeof(std::uint32_t),
        static_cast<std::size_t>(triangle_info.shape[0])));
    if (vertex_buffer == nullptr || index_buffer == nullptr) {
      rtcReleaseGeometry(geometry);
      rtcReleaseScene(object_scene);
      throw std::runtime_error("failed to allocate Embree geometry buffers");
    }
    std::memcpy(vertex_buffer, vertex_info.ptr,
                static_cast<std::size_t>(vertex_info.size) * sizeof(float));
    std::memcpy(index_buffer, triangle_info.ptr,
                static_cast<std::size_t>(triangle_info.size) *
                    sizeof(std::uint32_t));
    rtcCommitGeometry(geometry);
    throw_device_error(device_, "rtcCommitGeometry");
    const std::uint32_t object_geometry_id =
        rtcAttachGeometry(object_scene, geometry);
    rtcReleaseGeometry(geometry);
    if (object_geometry_id == RTC_INVALID_GEOMETRY_ID) {
      rtcReleaseScene(object_scene);
      throw std::runtime_error("failed to attach Embree object geometry");
    }
    rtcCommitScene(object_scene);
    throw_device_error(device_, "rtcCommitScene(object)");

    RTCGeometry instance = rtcNewGeometry(device_, RTC_GEOMETRY_TYPE_INSTANCE);
    if (instance == nullptr) {
      rtcReleaseScene(object_scene);
      throw std::runtime_error("failed to create Embree instance geometry");
    }
    rtcSetGeometryBuildQuality(instance, RTC_BUILD_QUALITY_REFIT);
    rtcSetGeometryInstancedScene(instance, object_scene);
    constexpr std::array<float, 16> identity = {
        1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F,
        0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F};
    rtcSetGeometryTransform(instance, 0, RTC_FORMAT_FLOAT4X4_COLUMN_MAJOR,
                            identity.data());
    rtcCommitGeometry(instance);
    throw_device_error(device_, "rtcCommitGeometry(instance)");
    const std::uint32_t geometry_id = rtcAttachGeometry(scene_, instance);
    rtcReleaseGeometry(instance);
    if (geometry_id == RTC_INVALID_GEOMETRY_ID) {
      rtcReleaseScene(object_scene);
      throw std::runtime_error("failed to attach Embree instance");
    }
    GeometryRecord record;
    record.material_index = material_index;
    record.mode = mode;
    record.thickness_m = thickness_m;
    record.grazing_cosine_floor = grazing_cosine_floor;
    record.object_scene = object_scene;
    geometry_records_.emplace(geometry_id, std::move(record));
    material_count_ = std::max<std::size_t>(
        material_count_, static_cast<std::size_t>(material_index) + 1);
    committed_ = false;
    return geometry_id;
  }

  void update_instance_transform(
      std::uint32_t geometry_id,
      const py::array_t<double, py::array::c_style | py::array::forcecast>
          &transform) {
    const auto transform_info = transform.request();
    if (transform_info.ndim != 2 || transform_info.shape[0] != 4 ||
        transform_info.shape[1] != 4) {
      throw std::invalid_argument("transform must have shape (4, 4)");
    }
    const auto *matrix = static_cast<const double *>(transform_info.ptr);
    for (py::ssize_t index = 0; index < transform_info.size; ++index) {
      if (!std::isfinite(matrix[index])) {
        throw std::invalid_argument("transform must contain finite values");
      }
    }
    std::unique_lock lock(mutex_);
    auto record_iterator = geometry_records_.find(geometry_id);
    if (record_iterator == geometry_records_.end()) {
      throw std::out_of_range("unknown Embree geometry ID");
    }
    RTCGeometry geometry = rtcGetGeometry(scene_, geometry_id);
    if (geometry == nullptr) {
      throw std::runtime_error("Embree geometry is detached");
    }
    std::array<float, 16> column_major{};
    for (std::size_t row = 0; row < 4; ++row) {
      for (std::size_t column = 0; column < 4; ++column) {
        column_major[column * 4 + row] =
            static_cast<float>(matrix[row * 4 + column]);
      }
    }
    rtcSetGeometryTransform(geometry, 0, RTC_FORMAT_FLOAT4X4_COLUMN_MAJOR,
                            column_major.data());
    rtcCommitGeometry(geometry);
    throw_device_error(device_, "update_instance_transform");
    committed_ = false;
  }

  void remove_geometry(std::uint32_t geometry_id) {
    std::unique_lock lock(mutex_);
    auto record = geometry_records_.find(geometry_id);
    if (record == geometry_records_.end()) {
      throw std::out_of_range("unknown Embree geometry ID");
    }
    rtcDetachGeometry(scene_, geometry_id);
    throw_device_error(device_, "rtcDetachGeometry");
    if (record->second.object_scene != nullptr) {
      rtcReleaseScene(record->second.object_scene);
    }
    geometry_records_.erase(record);
    committed_ = false;
  }

  void commit() {
    std::unique_lock lock(mutex_);
    rtcCommitScene(scene_);
    throw_device_error(device_, "rtcCommitScene");
    committed_ = true;
    ++revision_;
  }

  [[nodiscard]] std::uint64_t revision() const {
    std::shared_lock lock(mutex_);
    return revision_;
  }

  [[nodiscard]] std::size_t geometry_count() const {
    std::shared_lock lock(mutex_);
    return geometry_records_.size();
  }

  [[nodiscard]] std::size_t material_count() const {
    std::shared_lock lock(mutex_);
    return material_count_;
  }

  py::array_t<double> trace_path_lengths(
      const py::array_t<double, py::array::c_style | py::array::forcecast>
          &origins_m,
      const py::array_t<double, py::array::c_style | py::array::forcecast>
          &targets_m) const {
    const auto input = validate_segments(origins_m, targets_m);
    std::shared_lock lock(mutex_);
    require_committed();
    const std::size_t material_count = material_count_;
    py::array_t<double> result(
        {static_cast<py::ssize_t>(input.ray_count),
         static_cast<py::ssize_t>(material_count)});
    auto result_info = result.request();
    auto *output = static_cast<double *>(result_info.ptr);
    {
      py::gil_scoped_release release;
      const std::size_t packet_count =
          (input.ray_count + packet_width_ - 1) / packet_width_;
      parallel_for(packet_count, [&](std::size_t packet_index) {
        const std::size_t first_ray = packet_index * packet_width_;
        trace_packet8(input.origins, input.targets,
                      first_ray, input.ray_count,
                      output + first_ray * material_count);
      });
      throw_device_error(device_, "rtcIntersect8");
    }
    return result;
  }

  py::array_t<double> trace_transmission(
      const py::array_t<double, py::array::c_style | py::array::forcecast>
          &origins_m,
      const py::array_t<double, py::array::c_style | py::array::forcecast>
          &targets_m,
      const py::array_t<double, py::array::c_style | py::array::forcecast>
          &attenuation_per_m) const {
    const auto input = validate_segments(origins_m, targets_m);
    std::shared_lock lock(mutex_);
    require_committed();
    const std::size_t material_count = material_count_;
    const auto attenuation_info = attenuation_per_m.request();
    if (attenuation_info.ndim != 2 || attenuation_info.shape[1] == 0 ||
        attenuation_info.shape[0] < static_cast<py::ssize_t>(material_count)) {
      throw std::invalid_argument(
          "attenuation_per_m must have shape (material, energy_bin)");
    }
    const auto energy_count =
        static_cast<std::size_t>(attenuation_info.shape[1]);
    const auto *attenuation =
        static_cast<const double *>(attenuation_info.ptr);
    py::array_t<double> result(
        {static_cast<py::ssize_t>(input.ray_count),
         static_cast<py::ssize_t>(energy_count)});
    auto result_info = result.request();
    auto *output = static_cast<double *>(result_info.ptr);
    {
      py::gil_scoped_release release;
      const std::size_t packet_count =
          (input.ray_count + packet_width_ - 1) / packet_width_;
      parallel_for(packet_count, [&](std::size_t packet_index) {
        const std::size_t first_ray = packet_index * packet_width_;
        std::vector<double> path_lengths(packet_width_ * material_count, 0.0);
        trace_packet8(input.origins, input.targets, first_ray,
                      input.ray_count, path_lengths.data(), material_count);
        const std::size_t lanes =
            std::min(packet_width_, input.ray_count - first_ray);
        for (std::size_t lane = 0; lane < lanes; ++lane) {
          for (std::size_t energy = 0; energy < energy_count; ++energy) {
            double optical_depth = 0.0;
            for (std::size_t material = 0; material < material_count;
                 ++material) {
              optical_depth +=
                  path_lengths[lane * material_count + material] *
                  attenuation[material * energy_count + energy];
            }
            output[(first_ray + lane) * energy_count + energy] =
                std::exp(-std::min(optical_depth, 700.0));
          }
        }
      });
      throw_device_error(device_, "rtcIntersect8");
    }
    return result;
  }

private:
  struct SegmentInput {
    const double *origins;
    const double *targets;
    std::size_t ray_count;
  };

  SegmentInput validate_segments(
      const py::array_t<double, py::array::c_style | py::array::forcecast>
          &origins_m,
      const py::array_t<double, py::array::c_style | py::array::forcecast>
          &targets_m) const {
    const auto origin_info = origins_m.request();
    const auto target_info = targets_m.request();
    require_shape(origin_info, 3, "origins_m");
    require_shape(target_info, 3, "targets_m");
    if (origin_info.shape[0] != target_info.shape[0]) {
      throw std::invalid_argument(
          "origins_m and targets_m must have equal rows");
    }
    return {static_cast<const double *>(origin_info.ptr),
            static_cast<const double *>(target_info.ptr),
            static_cast<std::size_t>(origin_info.shape[0])};
  }

  void require_committed() const {
    if (!committed_) {
      throw std::logic_error("commit() must be called before tracing");
    }
  }

  void trace_packet8(const double *origin, const double *target,
                     std::size_t first_ray, std::size_t ray_count,
                     double *material_path_m,
                     std::size_t output_stride = 0) const {
    output_stride = output_stride == 0 ? material_count_ : output_stride;
    std::array<float, packet_width_> direction_x{};
    std::array<float, packet_width_> direction_y{};
    std::array<float, packet_width_> direction_z{};
    std::array<float, packet_width_> endpoint_epsilon{};
    std::array<float, packet_width_> segment_end{};
    std::array<float, packet_width_> next_tnear{};
    std::array<int, packet_width_> active{};
    std::array<std::unordered_map<std::uint32_t, float>, packet_width_>
        solid_entry_distance;

    for (std::size_t lane = 0; lane < packet_width_; ++lane) {
      const std::size_t ray_index = first_ray + lane;
      if (ray_index >= ray_count) {
        continue;
      }
      double *paths = material_path_m + lane * output_stride;
      std::fill(paths, paths + material_count_, 0.0);
      const double delta_x = target[3 * ray_index] - origin[3 * ray_index];
      const double delta_y =
          target[3 * ray_index + 1] - origin[3 * ray_index + 1];
      const double delta_z =
          target[3 * ray_index + 2] - origin[3 * ray_index + 2];
      const double length_m =
          std::sqrt(delta_x * delta_x + delta_y * delta_y +
                    delta_z * delta_z);
      if (!(length_m > 0.0)) {
        continue;
      }
      direction_x[lane] = static_cast<float>(delta_x / length_m);
      direction_y[lane] = static_cast<float>(delta_y / length_m);
      direction_z[lane] = static_cast<float>(delta_z / length_m);
      endpoint_epsilon[lane] =
          static_cast<float>(std::max(1.0e-6, length_m * 1.0e-6));
      segment_end[lane] =
          static_cast<float>(length_m) - endpoint_epsilon[lane];
      next_tnear[lane] = endpoint_epsilon[lane];
      active[lane] = segment_end[lane] > endpoint_epsilon[lane] ? -1 : 0;
    }

    for (std::size_t hit_count = 0; hit_count < 4096; ++hit_count) {
      alignas(32) RTCRayHit8 ray_hit{};
      std::array<int, packet_width_> valid{};
      bool any_active = false;
      for (std::size_t lane = 0; lane < packet_width_; ++lane) {
        valid[lane] = active[lane];
        any_active = any_active || active[lane] != 0;
        const std::size_t ray_index =
            std::min(first_ray + lane, ray_count == 0 ? 0 : ray_count - 1);
        ray_hit.ray.org_x[lane] = static_cast<float>(origin[3 * ray_index]);
        ray_hit.ray.org_y[lane] =
            static_cast<float>(origin[3 * ray_index + 1]);
        ray_hit.ray.org_z[lane] =
            static_cast<float>(origin[3 * ray_index + 2]);
        ray_hit.ray.dir_x[lane] = direction_x[lane];
        ray_hit.ray.dir_y[lane] = direction_y[lane];
        ray_hit.ray.dir_z[lane] = direction_z[lane];
        ray_hit.ray.tnear[lane] = next_tnear[lane];
        ray_hit.ray.tfar[lane] = segment_end[lane];
        ray_hit.ray.time[lane] = 0.0F;
        ray_hit.ray.mask[lane] = 0xFFFFFFFFU;
        ray_hit.ray.id[lane] = static_cast<unsigned int>(first_ray + lane);
        ray_hit.ray.flags[lane] = 0U;
        ray_hit.hit.geomID[lane] = RTC_INVALID_GEOMETRY_ID;
        ray_hit.hit.primID[lane] = RTC_INVALID_GEOMETRY_ID;
        for (unsigned int level = 0; level < RTC_MAX_INSTANCE_LEVEL_COUNT;
             ++level) {
          ray_hit.hit.instID[level][lane] = RTC_INVALID_GEOMETRY_ID;
        }
      }
      if (!any_active) {
        break;
      }
      RTCIntersectArguments arguments;
      rtcInitIntersectArguments(&arguments);
      rtcIntersect8(valid.data(), scene_, &ray_hit, &arguments);

      for (std::size_t lane = 0; lane < packet_width_; ++lane) {
        if (active[lane] == 0) {
          continue;
        }
        if (ray_hit.hit.geomID[lane] == RTC_INVALID_GEOMETRY_ID) {
          active[lane] = 0;
          continue;
        }
        const std::uint32_t geometry_id =
            ray_hit.hit.instID[0][lane] == RTC_INVALID_GEOMETRY_ID
                ? ray_hit.hit.geomID[lane]
                : ray_hit.hit.instID[0][lane];
        const auto record_iterator = geometry_records_.find(geometry_id);
        if (record_iterator == geometry_records_.end()) {
          active[lane] = 0;
          continue;
        }
        const GeometryRecord &record = record_iterator->second;
        const float hit_distance = ray_hit.ray.tfar[lane];
        const double normal_length = std::sqrt(
            static_cast<double>(ray_hit.hit.Ng_x[lane]) *
                ray_hit.hit.Ng_x[lane] +
            static_cast<double>(ray_hit.hit.Ng_y[lane]) *
                ray_hit.hit.Ng_y[lane] +
            static_cast<double>(ray_hit.hit.Ng_z[lane]) *
                ray_hit.hit.Ng_z[lane]);
        const double direction_dot_normal =
            normal_length > 0.0
                ? (direction_x[lane] * ray_hit.hit.Ng_x[lane] +
                   direction_y[lane] * ray_hit.hit.Ng_y[lane] +
                   direction_z[lane] * ray_hit.hit.Ng_z[lane]) /
                      normal_length
                : 0.0;
        double *paths = material_path_m + lane * output_stride;
        if (record.mode == GeometryMode::ThinSheet) {
          const double incidence =
              std::max(std::abs(direction_dot_normal),
                       record.grazing_cosine_floor);
          paths[record.material_index] += record.thickness_m / incidence;
        } else {
          auto &entries = solid_entry_distance[lane];
          const auto entry_iterator = entries.find(geometry_id);
          if (direction_dot_normal > 0.0) {
            const float entry = entry_iterator == entries.end()
                                    ? endpoint_epsilon[lane]
                                    : entry_iterator->second;
            paths[record.material_index] +=
                std::max(0.0, static_cast<double>(hit_distance - entry));
            if (entry_iterator != entries.end()) {
              entries.erase(entry_iterator);
            }
          } else if (entry_iterator == entries.end()) {
            entries.emplace(geometry_id, hit_distance);
          }
        }
        const float hit_epsilon =
            std::max(1.0e-5F, 1.0e-5F * std::abs(hit_distance));
        next_tnear[lane] = hit_distance + hit_epsilon;
        if (next_tnear[lane] >= segment_end[lane]) {
          active[lane] = 0;
        }
      }
    }
    for (std::size_t lane = 0; lane < packet_width_; ++lane) {
      double *paths = material_path_m + lane * output_stride;
      for (const auto &[geometry_id, entry] : solid_entry_distance[lane]) {
        const auto record_iterator = geometry_records_.find(geometry_id);
        if (record_iterator != geometry_records_.end()) {
          paths[record_iterator->second.material_index] += std::max(
              0.0, static_cast<double>(segment_end[lane] - entry));
        }
      }
    }
  }

  static constexpr std::size_t packet_width_ = 8;

  RTCDevice device_;
  RTCScene scene_;
  std::unordered_map<std::uint32_t, GeometryRecord> geometry_records_;
  std::size_t material_count_ = 0;
  bool committed_ = false;
  std::uint64_t revision_ = 0;
  mutable std::shared_mutex mutex_;
};

} // namespace

PYBIND11_MODULE(_radcounter_embree, module) {
  module.doc() = "Embree 4 dynamic segment attenuation backend for RadCounterSim";
  module.def("embree_version", []() { return RTC_VERSION_STRING; });
  py::class_<EmbreeScene>(module, "Scene")
      .def(py::init<>())
      .def("add_triangle_mesh", &EmbreeScene::add_triangle_mesh,
           py::arg("vertices_m"), py::arg("triangles"),
           py::arg("material_index"), py::arg("geometry_mode") = "solid",
           py::arg("thickness_m") = 0.0,
           py::arg("grazing_cosine_floor") = 0.05)
      .def("update_instance_transform", &EmbreeScene::update_instance_transform,
           py::arg("geometry_id"), py::arg("transform"))
      .def("remove_geometry", &EmbreeScene::remove_geometry,
           py::arg("geometry_id"))
      .def("commit", &EmbreeScene::commit)
      .def_property_readonly("revision", &EmbreeScene::revision)
      .def_property_readonly("geometry_count", &EmbreeScene::geometry_count)
      .def_property_readonly("material_count", &EmbreeScene::material_count)
      .def_property_readonly("packet_width",
                             [](const EmbreeScene &) { return 8; })
      .def("trace_path_lengths", &EmbreeScene::trace_path_lengths,
           py::arg("origins_m"), py::arg("targets_m"))
      .def("trace_transmission", &EmbreeScene::trace_transmission,
           py::arg("origins_m"), py::arg("targets_m"),
           py::arg("attenuation_per_m"));
  module.attr("EmbreeTransportScene") = module.attr("Scene");
}
