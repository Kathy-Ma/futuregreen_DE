import os
folder_path = r"C:\Users\meeri\Downloads\Metal\steel_food_cans\default"
for i in range(1, 251):
    old_name = f"image_{i}.png"
    new_i = i + 1250
    new_name = f"metal_{new_i}.png"
    old_file = os.path.join(folder_path, old_name)
    new_file = os.path.join(folder_path, new_name)
    if os.path.exists(old_file):
        os.rename(old_file, new_file)
        print(f"Renamed: {old_name} -> {new_name}")
    else:
        print(f"Skipped: {old_name} (File not found)")
print("Done!")