import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer


# st.title('Hello World!')

# st.write('This is a simple Streamlit app.')

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        # Perform any image processing operations here
        return frame


def main():
    st.title("OpenCV with Streamlit-WebRTC")

    webrtc_ctx = webrtc_streamer(
        key="example",
        video_transformer_factory=VideoTransformer,
        async_transform=True,
    )

    if webrtc_ctx.video_transformer:
        st.write("Webcam is running")
