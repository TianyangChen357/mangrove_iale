from landsatxplore.earthexplorer import EarthExplorer

# Your USGS Earth Explorer credentials
username = "tchen19"
password = "Cty15830061892"

# Input file with entity IDs
input_file = "scene_list_entity_id.txt"

# Output directory for downloads
output_dir = "landsat_downloads"

# Read entity IDs from file
with open(input_file, "r") as f:
    entity_ids = [line.strip() for line in f.readlines() if line.strip()]

# Initialize EarthExplorer for downloading
ee = EarthExplorer(username, password)

# Download each scene
for entity_id in entity_ids:
    print(f"Downloading {entity_id}...")
    try:
        ee.download(entity_id, output_dir=output_dir)
        print(f"✅ Successfully downloaded: {entity_id}")
    except Exception as e:
        print(f"❌ Error downloading {entity_id}: {e}")

# Logout
ee.logout()
print("Download complete!")
