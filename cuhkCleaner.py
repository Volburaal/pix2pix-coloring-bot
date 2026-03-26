import os
import shutil

photos_dir = "data/cuhk/photos"
sketches_dir = "data/cuhk/sketches"

extras_photos = "data/cuhk/Extras/photos"
extras_sketches = "data/cuhk/Extras/sketches"

os.makedirs(extras_photos, exist_ok=True)
os.makedirs(extras_sketches, exist_ok=True)

# Helper to remove suffix like -01 or -sz1
def normalize_name(filename):
    name, ext = os.path.splitext(filename)
    # name = name.replace("-01", "")
    name = name.replace("-sz1", "")
    return name + ext

# Process photos
for photo_file in os.listdir(photos_dir):
    photo_path = os.path.join(photos_dir, photo_file)

    if not os.path.isfile(photo_path):
        continue

    name, ext = os.path.splitext(photo_file)

    # Case 1: exact same name exists
    sketch_file = photo_file
    sketch_path = os.path.join(sketches_dir, sketch_file)

    # Case 2: try adding -sz1
    if not os.path.exists(sketch_path):
        sketch_file = name + "-sz1" + ext
        sketch_path = os.path.join(sketches_dir, sketch_file)

    # MATCH FOUND
    if os.path.exists(sketch_path):
        new_name = normalize_name(photo_file)

        new_photo_path = os.path.join(photos_dir, new_name)
        new_sketch_path = os.path.join(sketches_dir, new_name)

        os.rename(photo_path, new_photo_path)
        os.rename(sketch_path, new_sketch_path)

    else:
        # NO MATCH → move photo
        shutil.move(photo_path, os.path.join(extras_photos, photo_file))

# Now handle sketches that were never matched
for sketch_file in os.listdir(sketches_dir):
    sketch_path = os.path.join(sketches_dir, sketch_file)

    if not os.path.isfile(sketch_path):
        continue

    normalized = normalize_name(sketch_file)
    corresponding_photo = os.path.join(photos_dir, normalized)

    if not os.path.exists(corresponding_photo):
        shutil.move(sketch_path, os.path.join(extras_sketches, sketch_file))

print("Done processing dataset.")