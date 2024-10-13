import streamlit as st

# Toggle selection of an image
def toggle_select(image_info):
    if image_info in st.session_state.selected_images:
        st.session_state.selected_images.remove(image_info)
    else:
        st.session_state.selected_images.append(image_info)

# Toggle deletion of an image
def toggle_delete(image_path):
    if image_path in st.session_state.marked_for_deletion:
        st.session_state.marked_for_deletion.remove(image_path)
    else:
        st.session_state.marked_for_deletion.add(image_path)

# Filter out deleted images from the results
def filter_out_deleted(image_paths):
    return [path for path in image_paths if path not in st.session_state.deleted_images]

# Delete all loaded images
def delete_all_loaded_images(image_paths):
    st.session_state.deleted_images.update(image_paths)
