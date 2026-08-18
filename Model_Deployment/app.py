# ============================================================
# TRAFFIC SIGN AI
# Deep Learning Traffic Sign Recognition
# ============================================================

import os

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Traffic Sign AI",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CONFIGURATION
# ============================================================

IMG_SIZE = (64, 64)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "model",
    "model1.keras"
)

MODEL_PATH = os.path.abspath(MODEL_PATH)


# ============================================================
# TRAFFIC SIGN NAMES
# GTSRB - 43 CLASSES
# ============================================================

CLASS_NAMES = [
    "Speed limit 20 km/h",
    "Speed limit 30 km/h",
    "Speed limit 50 km/h",
    "Speed limit 60 km/h",
    "Speed limit 70 km/h",
    "Speed limit 80 km/h",
    "End of speed limit 80 km/h",
    "Speed limit 100 km/h",
    "Speed limit 120 km/h",
    "No passing",
    "No passing for vehicles over 3.5 tons",
    "Right-of-way at next intersection",
    "Priority road",
    "Yield",
    "Stop",
    "No vehicles",
    "Vehicles over 3.5 tons prohibited",
    "No entry",
    "General caution",
    "Dangerous curve left",
    "Dangerous curve right",
    "Double curve",
    "Bumpy road",
    "Slippery road",
    "Road narrows on the right",
    "Road work",
    "Traffic signals",
    "Pedestrians",
    "Children crossing",
    "Bicycles crossing",
    "Beware of ice/snow",
    "Wild animals crossing",
    "End of all speed and passing limits",
    "Turn right ahead",
    "Turn left ahead",
    "Ahead only",
    "Go straight or right",
    "Go straight or left",
    "Keep right",
    "Keep left",
    "Roundabout mandatory",
    "End of no passing",
    "End of no passing by vehicles over 3.5 tons"
]


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    """
    Load the trained traffic sign recognition model.
    """

    if not os.path.exists(MODEL_PATH):
        st.error(
            "⚠️ The trained model could not be found.\n\n"
            f"Expected location:\n{MODEL_PATH}"
        )

        st.stop()

    try:
        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )

        return model

    except Exception as error:

        st.error(
            "⚠️ The model could not be loaded."
        )

        st.exception(error)

        st.stop()


model = load_model()


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_traffic_sign(image):
    """
    Prepare an uploaded image and make a prediction.

    Returns:
        predicted_class
        confidence
        other_predictions
    """

    # --------------------------------------------------------
    # Resize image
    # --------------------------------------------------------

    image = image.convert("RGB")

    image_resized = image.resize(IMG_SIZE)


    # --------------------------------------------------------
    # Convert image to NumPy array
    # --------------------------------------------------------

    image_array = np.asarray(
        image_resized,
        dtype=np.float32
    )


    # --------------------------------------------------------
    # Normalize pixel values
    # --------------------------------------------------------

    image_array = image_array / 255.0


    # --------------------------------------------------------
    # Add batch dimension
    # Shape becomes:
    # (1, 64, 64, 3)
    # --------------------------------------------------------

    image_batch = np.expand_dims(
        image_array,
        axis=0
    )


    # --------------------------------------------------------
    # Model prediction
    # --------------------------------------------------------

    predictions = model.predict(
        image_batch,
        verbose=0
    )[0]


    # --------------------------------------------------------
    # Predicted class
    # --------------------------------------------------------

    predicted_class = int(
        np.argmax(predictions)
    )


    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    confidence = float(
        predictions[predicted_class]
    )


    # --------------------------------------------------------
    # Find two other possible matches
    # --------------------------------------------------------

    top_indices = np.argsort(
        predictions
    )[::-1]


    other_predictions = []


    for index in top_indices:

        index = int(index)


        # Do not repeat the main prediction
        if index == predicted_class:
            continue


        other_predictions.append(
            {
                "name": CLASS_NAMES[index],
                "confidence": float(
                    predictions[index]
                )
            }
        )


        # Only keep two alternatives
        if len(other_predictions) == 2:
            break


    return (
        predicted_class,
        confidence,
        other_predictions
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "# 🚦 Traffic Sign AI"
    )

    st.write(
        "An AI-powered application that recognizes "
        "traffic signs from images."
    )

    st.divider()

    st.subheader("🤖 About the AI")

    st.write(
        "The application uses a Convolutional Neural Network "
        "(CNN) trained to recognize traffic signs."
    )

    st.write(
        "**43 different traffic sign types** can be recognized."
    )

    st.write(
        "**95.14% test accuracy** was achieved during model testing."
    )

    st.divider()

    st.subheader("📌 How to use")

    st.write(
        "1. Upload a traffic sign image."
    )

    st.write(
        "2. Wait for the AI to analyze it."
    )

    st.write(
        "3. View the predicted sign."
    )

    st.write(
        "4. Check how confident the AI is."
    )

    st.divider()

    st.caption(
        "Built with Python • TensorFlow • Streamlit"
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.title("🚦 Traffic Sign AI")

st.markdown(
    "### Intelligent traffic sign recognition using Deep Learning"
)

st.write(
    "Upload a clear image of a traffic sign and let the AI "
    "identify what it is."
)


# ============================================================
# INFORMATION BOX
# ============================================================

st.info(
    "💡 For the best result, upload a clear image where the "
    "traffic sign is visible and not heavily blurred."
)


# ============================================================
# UPLOAD SECTION
# ============================================================

st.subheader(" Upload Traffic Sign")

uploaded_file = st.file_uploader(
    "Choose a traffic sign image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ],
    help="Upload a JPG, JPEG or PNG image."
)


# ============================================================
# PROCESS IMAGE
# ============================================================

if uploaded_file is not None:

    try:

        # ----------------------------------------------------
        # Open uploaded image
        # ----------------------------------------------------

        image = Image.open(
            uploaded_file
        ).convert("RGB")


        # ----------------------------------------------------
        # Display image and prediction side-by-side
        # ----------------------------------------------------

        image_column, prediction_column = st.columns(
            [1, 1],
            gap="large"
        )


        # ====================================================
        # INPUT IMAGE
        # ====================================================

        with image_column:

            st.subheader("🖼️ Input Image")

            st.image(
                image,
                caption="Uploaded Traffic Sign",
                width="stretch"
            )


        # ====================================================
        # AI PREDICTION
        # ====================================================

        with prediction_column:

            st.subheader("🤖 AI Prediction")

            with st.spinner(
                "AI is analyzing the traffic sign..."
            ):

                (
                    predicted_class,
                    confidence,
                    other_predictions
                ) = predict_traffic_sign(
                    image
                )


            predicted_name = CLASS_NAMES[
                predicted_class
            ]


            # ------------------------------------------------
            # Main prediction
            # ------------------------------------------------

            st.success(
                f" **{predicted_name}**"
            )


            # ------------------------------------------------
            # Confidence
            # ------------------------------------------------

            st.metric(
                "Model Confidence",
                f"{confidence * 100:.2f}%"
            )


            st.progress(
                min(max(confidence, 0.0), 1.0)
            )


            # ------------------------------------------------
            # Simple explanation
            # ------------------------------------------------

            if confidence >= 0.90:

                st.success(
                    "The AI is highly confident about this result."
                )

            elif confidence >= 0.70:

                st.warning(
                    "The AI is reasonably confident, but "
                    "the image may contain some uncertainty."
                )

            else:

                st.warning(
                    "The AI is not very confident. "
                    "Try uploading a clearer image."
                )


    except Exception as error:

        st.error(
            "⚠️ Something went wrong while processing "
            "the image."
        )

        st.exception(error)


    # ========================================================
    # OTHER POSSIBLE MATCHES
    # ========================================================

    st.divider()

    st.subheader(
        " Other Possible Matches"
    )

    st.caption(
        "These are two other signs the AI considered possible."
    )


    if other_predictions:

        for item in other_predictions:

            col1, col2 = st.columns(
                [3, 1]
            )

            with col1:

                st.write(
                    f"**{item['name']}**"
                )

            with col2:

                st.write(
                    f"{item['confidence'] * 100:.1f}%"
                )

    else:

        st.caption(
            "No other possible matches available."
        )


    # ========================================================
    # HOW THE AI WORKS
    # ========================================================

    st.divider()

    st.subheader(
        " How does the AI work?"
    )

    with st.expander(
        "Learn how your image is analyzed"
    ):

        st.write(
            " **1. You upload an image**"
        )

        st.write(
            "The application receives your traffic sign image."
        )

        st.write(
            " **2. The image is prepared**"
        )

        st.write(
            "The image is resized and prepared so the AI "
            "can understand it."
        )

        st.write(
            " **3. The AI looks at the image**"
        )

        st.write(
            "The deep learning model looks for important "
            "patterns, shapes and visual features."
        )

        st.write(
            " **4. The AI makes a prediction**"
        )

        st.write(
            "The model compares the image with what it "
            "learned during training."
        )

        st.write(
            " **5. You get the result**"
        )

        st.write(
            "The application shows the traffic sign that "
            "the AI thinks is the best match and how "
            "confident it is."
        )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.divider()

st.header(
    "How Good Is the AI?"
)

st.write(
    "The model was tested using traffic sign images "
    "that were not used during training."
)


performance_col1, performance_col2 = st.columns(
    2
)


with performance_col1:

    st.metric(
        "🎯 Recognition Accuracy",
        "95.14%"
    )

    st.caption(
        "The AI correctly identifies about 95 out of "
        "100 test images."
    )


with performance_col2:

    st.metric(
        "🚦 Traffic Signs Recognized",
        "43"
    )

    st.caption(
        "The AI can recognize 43 different types of "
        "traffic signs."
    )


st.info(
    "💡 In simple words: during testing, the AI correctly "
    "identified traffic signs about 95% of the time."
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🚦 Traffic Sign AI • Deep Learning • Computer Vision"
)

st.caption(
    "This application is for demonstration and educational "
    "purposes. Always follow official road signs and traffic rules."
)