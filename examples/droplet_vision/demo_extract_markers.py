import os

from chemobot_tools.droplet_labelling.extract_markers import extract_markers_from_folder, save_marker_imgs


if __name__ == '__main__':

    for fname in ['droplet', 'empty']:

        labelled_folder = os.path.join('labelled', fname)
        extracted_folder = os.path.join('extracted', fname)

        all_maker_imgs = extract_markers_from_folder(labelled_folder, windows_size=0, frame_stepping=1, verbose=True)

        save_marker_imgs(all_maker_imgs, extracted_folder)
