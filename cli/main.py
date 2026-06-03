from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(help="LiDAR point cloud classification via multi-agent feedback loop.")


@app.command()
def classify(
    input_file: Path = typer.Argument(..., help="LAS/LAZ/PLY input file"),
    output_dir: Path = typer.Option(Path("./output"), help="Output directory"),
    max_range: float = typer.Option(100.0, help="Max range filter (meters)"),
    min_range: float = typer.Option(0.05, help="Min range filter (meters)"),
    max_iterations: int = typer.Option(5, help="Max feedback loop iterations"),
    min_iterations: int = typer.Option(1, help="Min iterations before early stopping"),
    export_dxf: bool = typer.Option(False, help="Export DXF floor plan"),
    export_svg: bool = typer.Option(False, help="Export SVG floor plan"),
    export_geojson: bool = typer.Option(False, help="Export GeoJSON floor plan"),
    save_intermediate: bool = typer.Option(True, help="Save per-iteration results"),
    voxel_size: Optional[float] = typer.Option(None, help="Voxel downsampling size (meters)"),
) -> None:
    """Full pipeline: preprocess → classify (iterative feedback) → export."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from pointcloud import io as pc_io
    from pointcloud.preprocessing import PreprocessingPipeline, PreprocessingConfig
    from pointcloud.floorplan import FloorPlanExtractor, FloorPlanExporter
    from agents.base import get_anthropic_client
    from agents.classifier_agent import ClassifierAgent
    from agents.validator_agent import ValidatorAgent
    from orchestrator.session import ClassificationSession
    from orchestrator.feedback_loop import IterativeFeedbackOrchestrator, FeedbackLoopConfig

    output_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"Loading {input_file}...")
    cloud = pc_io.load(input_file)
    typer.echo(f"  {cloud.n_points:,} points loaded.")

    typer.echo("Preprocessing...")
    cfg = PreprocessingConfig(
        max_range_m=max_range,
        min_range_m=min_range,
        voxel_downsample_m=voxel_size,
    )
    pipeline = PreprocessingPipeline(cfg)
    cleaned, stats = pipeline.run(cloud)
    typer.echo(f"  Preprocessing stats: {stats}")

    client = get_anthropic_client()
    session = ClassificationSession(cloud=cleaned)
    classifier = ClassifierAgent(client=client, session=session)
    validator = ValidatorAgent(client=client, session=session)

    loop_cfg = FeedbackLoopConfig(
        max_iterations=max_iterations,
        min_iterations=min_iterations,
        save_intermediate=save_intermediate,
        intermediate_dir=str(output_dir / "runs"),
    )
    orchestrator = IterativeFeedbackOrchestrator(classifier, validator, loop_cfg)

    typer.echo("Running classification feedback loop...")
    result = orchestrator.run(cleaned)

    summary = result.summary()
    typer.echo(f"\nResult: {json.dumps(summary, indent=2)}")

    out_las = output_dir / "classified.las"
    pc_io.save_classified(result.best_classified_cloud.cloud, result.best_classified_cloud.labels, out_las)
    typer.echo(f"Saved: {out_las}")

    report_path = output_dir / "report.json"
    r = result.best_report
    with open(report_path, "w") as f:
        json.dump({
            "summary": summary,
            "feedback_text": r.metrics.feedback_text,
            "refinement_hints": r.refinement_hints,
            "label_coverage": r.metrics.label_coverage,
        }, f, indent=2)
    typer.echo(f"Saved: {report_path}")

    if export_dxf or export_svg or export_geojson:
        typer.echo("Extracting floor plan...")
        extractor = FloorPlanExtractor()
        floor_plan = extractor.extract(result.best_classified_cloud)
        exporter = FloorPlanExporter()

        if export_dxf:
            path = output_dir / "floorplan.dxf"
            exporter.to_dxf(floor_plan, path)
            typer.echo(f"Saved: {path}")
        if export_svg:
            path = output_dir / "floorplan.svg"
            exporter.to_svg(floor_plan, path)
            typer.echo(f"Saved: {path}")
        if export_geojson:
            path = output_dir / "floorplan.geojson"
            exporter.to_geojson(floor_plan, path)
            typer.echo(f"Saved: {path}")

    typer.echo("\nDone.")


@app.command()
def preprocess_only(
    input_file: Path = typer.Argument(...),
    output_file: Path = typer.Argument(...),
    max_range: float = typer.Option(100.0),
    min_range: float = typer.Option(0.05),
) -> None:
    """Run only the preprocessing pipeline and save the cleaned cloud."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from pointcloud import io as pc_io
    from pointcloud.preprocessing import PreprocessingPipeline, PreprocessingConfig

    cloud = pc_io.load(input_file)
    typer.echo(f"Loaded {cloud.n_points:,} points.")
    pipeline = PreprocessingPipeline(PreprocessingConfig(max_range_m=max_range, min_range_m=min_range))
    cleaned, stats = pipeline.run(cloud)
    typer.echo(f"Stats: {stats}")
    pc_io.save(cleaned, output_file)
    typer.echo(f"Saved to {output_file}")


@app.command()
def floorplan_from_classified(
    classified_las: Path = typer.Argument(..., help="Classified LAS file"),
    output_dir: Path = typer.Option(Path("./floorplan_out")),
    export_dxf: bool = typer.Option(True),
    export_svg: bool = typer.Option(False),
) -> None:
    """Extract a floor plan from an already-classified LAS file."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import laspy
    import numpy as np
    from pointcloud.models import PointCloud, ClassifiedCloud
    from pointcloud.floorplan import FloorPlanExtractor, FloorPlanExporter

    output_dir.mkdir(parents=True, exist_ok=True)
    las = laspy.read(str(classified_las))
    pts = np.stack([las.x, las.y, las.z], axis=1).astype(np.float64)
    labels = las.classification.astype(np.int8)
    confidence = np.ones(len(pts), dtype=np.float32)
    cloud = PointCloud(points=pts)
    cc = ClassifiedCloud(cloud=cloud, labels=labels, confidence=confidence)

    extractor = FloorPlanExtractor()
    floor_plan = extractor.extract(cc)
    exporter = FloorPlanExporter()

    if export_dxf:
        p = output_dir / "floorplan.dxf"
        exporter.to_dxf(floor_plan, p)
        typer.echo(f"Saved: {p}")
    if export_svg:
        p = output_dir / "floorplan.svg"
        exporter.to_svg(floor_plan, p)
        typer.echo(f"Saved: {p}")


if __name__ == "__main__":
    app()
