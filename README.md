# StudentSense — Student Success Predictor

A redesigned Streamlit interface for the original Random Forest student activity model. The original `model.joblib`, `meta.json`, dataset, and `train.py` are included. **The trained model and its evaluation metrics were not changed** by this redesign.

## Run in VS Code

1. Extract this ZIP and open the **`student-predictor`** folder in VS Code.
2. Open **Terminal → New Terminal**. Make sure the terminal is in the folder containing `app.py`.
3. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

   On Windows, if `python` is not recognized, use `py` instead.
4. Start the app:

   ```bash
   python -m streamlit run app.py
   ```

5. Open the local address shown in the terminal (usually `http://localhost:8501`).

## What's new

- Responsive dashboard with a dark teal hero header
- Eight original inputs arranged into **Academic engagement**, **Learning participation**, and **Platform interactions**
- **Analyze student profile**, **Reset to typical**, and **Clear all fields** controls
- Easy-to-read results with a model score visualization
- Global feature importance visualizations and contextual cautions
- No accuracy rates displayed in the app interface

## Important model notes

- The CSV target is named **`rating`**, with values **High** and **Low**. The original project displays these as **Pass** and **Fail**, but they are **not confirmed final course grades**. The UI makes this limitation explicit.
- Scores from `predict_proba` are raw model estimates, not guaranteed or necessarily calibrated probabilities.
- The training data contains augmented, duplicate profiles. See `train.py` and `meta.json` for the original evaluation details. Those evaluations were **not rerun** for this visual update.
- Don't use these predictions on their own for academic decisions about an individual student.

## Training or changing the model

The packaged trained model is ready to use; **you do not need to train again to run the app**. To retrain explicitly:

```bash
python train.py
```

This overwrites `model.joblib` and `meta.json`. Note that deployment results may differ if dependencies, data, or models change.

## Colab option

Use the included `student_predictor_colab_redesigned.ipynb` if you prefer Google Colab. Its `%%writefile app.py` cell contains the **new interface**, so it will not recreate the older design. The Colab version retrains the model as in the original notebook and serves the app through a localtunnel session.

## Streamlit Community Cloud

1. Push the extracted project files to a GitHub repository (include `app.py`, `model.joblib`, `meta.json`, and `requirements.txt`).
2. Open [Streamlit Community Cloud](https://share.streamlit.io/) and create a new app with `app.py` as the entry point.
3. The model file is around 32 MB, so keep it tracked and verify your hosting provider's repository and resource limits.

The app does **not** collect student names or save submitted profiles to a database. A prediction is held only in the current Streamlit session state.
