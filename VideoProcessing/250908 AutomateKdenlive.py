import xml.etree.ElementTree as ET
import subprocess
import os

def replace_clip_filename(filepath, new_filename, output_path=None):
    """
    Replace all occurrences of 'ClipFilenamePlaceHolder' in the kdenlive XML with new_filename.
    Save the modified XML to output_path or overwrite original if output_path is None.
    """
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()

        # Iterate over all elements and their text/attributes to replace the placeholder
        replaced_count = 0
        for elem in root.iter():
            # Replace in text if present
            if elem.text and "ClipFilenamePlaceHolder" in elem.text:
                elem.text = elem.text.replace("ClipFilenamePlaceHolder", new_filename)
                replaced_count += 1
            # Replace in attributes if present
            for attr in elem.attrib:
                if "ClipFilenamePlaceHolder" in elem.attrib[attr]:
                    elem.attrib[attr] = elem.attrib[attr].replace("ClipFilenamePlaceHolder", new_filename)
                    replaced_count += 1

        if replaced_count == 0:
            print("No occurrences of 'ClipFilenamePlaceHolder' found in the file.")
        else:
            print(f"Replaced {replaced_count} occurrences of 'ClipFilenamePlaceHolder' with '{new_filename}'.")

        # Determine where to save the modified file
        save_path = output_path if output_path else filepath
        tree.write(save_path, encoding="utf-8", xml_declaration=True)
        return save_path

    except ET.ParseError as e:
        print(f"Error parsing the Kdenlive file: {e}")
        return None
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return None

def open_kdenlive_with_app(filepath):
    kdenlive_path = r"C:\Program Files\Kdenlive\bin\kdenlive.exe"  # Adjust this path for your installation
    
    if not os.path.isfile(filepath):
        print(f"File not found: {filepath}")
        return
    
    if not os.path.isfile(kdenlive_path):
        print(f"Kdenlive executable not found at {kdenlive_path}. Please check the path.")
        return
    
    try:
        subprocess.run([kdenlive_path, filepath], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to open file with Kdenlive: {e}")

if __name__ == "__main__":
    original_kdenlive_path = r"C:\Users\jonat\Documents\GitHub\ECTE351AI\VideoProcessing\InputNormalOutputFiltered.kdenlive"
    new_clip_filename = "2025-08-26 13-17-13"
    
    modified_kdenlive_path = rf"C:\Users\jonat\Documents\GitHub\ECTE351AI\VideoProcessing\InputNormalOutputFiltered_{new_clip_filename}.kdenlive"    
    modified_file = replace_clip_filename(original_kdenlive_path, new_clip_filename, modified_kdenlive_path)
    if modified_file:
        open_kdenlive_with_app(modified_file)
