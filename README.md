# cis5810final

## 🧍 How to Test

### 0. Setup Environment
Make sure your environment satisfies requirement.txt

### 1. Prepare Inputs
Create a folder `FOLDER` to store your input images (ideally pictures taken by the user).
Here's a sample Ren compiled: [link](https://drive.google.com/file/d/1akAa_kisOb_4FkioHZY4VYBgbb_tRBbB/view?usp=sharing)

### 2. Segment Images
From the project root, run:

```
python segment/predict_and_segment_folder.py --images_fir FOLDER
```

This will segment each image and save results in an `out/` folder.

### 3. Process Segmented Images
Generate a wardrobe dataset with:

```
python recommend/wardrobe.py
```

This creates `data/wardrobe.csv` containing all segmented items.

### 4. Get Outfit Recommendations
Run:
```
python recommend/recommend.py
```
This recommends and displays an outfit based on your wardrobe.

***

# How to run frontend on local

We are using streamlit now, easier to run locally than full react application w/ some separate backend

To run:
- cd into root dir (cis5810final)
- Run `streamlit run streamlit_app/Home.py`

You'll need streamlit installed: `pip install streamlit`
