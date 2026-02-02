from ms_pipeline import MultispectralPipeline
import pprint

pipeline = MultispectralPipeline("./ms-images", "./data_layer/processed")
groups = pipeline.scan_and_group()

print(f"Found {len(groups)} groups.")
for gid, images in groups.items():

    if images:
        print(f"Group {gid}: {len(images)} images. Date Range: {images[0]['dt']} to {images[-1]['dt']}")
        print(f"  Avg Altitude: {pipeline._get_avg_altitude(images)}")
