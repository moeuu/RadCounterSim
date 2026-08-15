"""CLI for listing, validating, preparing, and activating system profiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from radcounter.core.environment import EnvironmentImportPipeline

from .system_profiles import (
    default_catalog_path,
    default_selection_path,
    load_active_selection,
    load_system_catalog,
    resolve_system_selection,
    save_active_selection,
)


def _selection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--catalog", type=Path, default=default_catalog_path())
    parser.add_argument("--profile")
    parser.add_argument("--environment")
    parser.add_argument("--robot-set")
    parser.add_argument("--detector-set")


def _resolve(args: argparse.Namespace):
    return resolve_system_selection(
        catalog_path=args.catalog,
        profile_id=args.profile,
        environment_id=args.environment,
        robot_set_id=args.robot_set,
        detector_set_id=args.detector_set,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Switch RadCounterSim environments, robots, and detectors"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="list catalog components and profiles")
    list_parser.add_argument("--catalog", type=Path, default=default_catalog_path())

    show_parser = subparsers.add_parser("show", help="show one fully resolved selection")
    _selection_arguments(show_parser)

    check_parser = subparsers.add_parser("check", help="validate paths and import prerequisites")
    _selection_arguments(check_parser)

    prepare_parser = subparsers.add_parser(
        "prepare", help="normalize the selected environment into the content-addressed cache"
    )
    _selection_arguments(prepare_parser)

    activate_parser = subparsers.add_parser(
        "activate", help="save the selection used by future radcounter-app launches"
    )
    _selection_arguments(activate_parser)
    activate_parser.add_argument("--selection-file", type=Path, default=default_selection_path())

    current_parser = subparsers.add_parser("current", help="show the active saved selection")
    current_parser.add_argument("--selection-file", type=Path, default=default_selection_path())

    args = parser.parse_args(argv)
    if args.command == "list":
        catalog_path, catalog = load_system_catalog(args.catalog)
        payload = {
            "catalog": str(catalog_path),
            "default_profile": catalog.default_profile,
            "profiles": {
                key: {
                    "display_name": value.display_name,
                    "environment": value.environment,
                    "robot_set": value.robot_set,
                    "detector_set": value.detector_set,
                    "application_mode": value.application_mode,
                }
                for key, value in catalog.profiles.items()
            },
            "environments": sorted(catalog.environments),
            "robot_sets": sorted(catalog.robot_sets),
            "detector_sets": sorted(catalog.detector_sets),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "current":
        print(json.dumps(load_active_selection(args.selection_file).as_dict(), indent=2))
        return 0

    selection = _resolve(args)
    if args.command == "show":
        print(json.dumps(selection.as_dict(), indent=2, sort_keys=True))
        return 0
    if args.command == "check":
        payload = selection.as_dict()
        if not selection.environment_ready:
            print(json.dumps({"valid": False, **payload}, indent=2, sort_keys=True))
            return 2
        print(json.dumps({"valid": True, **payload}, indent=2, sort_keys=True))
        return 0
    if args.command == "prepare":
        if not selection.environment_ready:
            source = selection.environment_source_path
            raise FileNotFoundError(
                f"selected environment is not ready: {source}; "
                f"{selection.environment_entry.setup_hint or 'provide the configured source file'}"
            )
        result = EnvironmentImportPipeline().import_environment(
            selection.environment_config,
            base_directory=selection.environment_descriptor_path.parent,
        )
        print(
            json.dumps(
                {
                    "prepared": True,
                    "profile": selection.profile_id,
                    "environment": selection.environment_id,
                    "manifest": str(result.manifest_path),
                    "vertices": result.scene.vertex_count,
                    "triangles": result.scene.triangle_count,
                    "bounds_m": result.scene.bounds_m,
                    "warnings": result.scene.warnings,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "activate":
        path = save_active_selection(
            args.selection_file,
            catalog_path=selection.catalog_path,
            profile_id=selection.profile_id,
            environment_id=args.environment,
            robot_set_id=args.robot_set,
            detector_set_id=args.detector_set,
        )
        print(
            json.dumps(
                {"activated": True, "selection_file": str(path), **selection.as_dict()},
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
