"""
Palakkad Land-Cover App — Streamlit / geemap port
==================================================
Python equivalent of the GEE-native App (02_landcover_app.js). Native GEE
Apps only run compiled JavaScript, so this uses geemap (folium-based) inside
Streamlit to reproduce the same map + legend + class-area + accuracy/kappa
panel, driven by the Earth Engine Python API.

Setup
-----
    pip install earthengine-api geemap streamlit
    earthengine authenticate          # one-time
    streamlit run 02_landcover_app.py

If you exported the classified image to an EE asset (see the commented-out
block at the end of 01_landcover_classification.py), swap the "CLASSIFY"
section below for `ee.Image('projects/<proj>/assets/palakkad_landcover_rf')`
so the app loads instantly instead of recomputing every run.
"""

import ee
import streamlit as st
import geemap.foliumap as geemap

EE_PROJECT = "your-gcp-project-id"  # <-- set this

st.set_page_config(layout="wide", page_title="Palakkad Land-Cover Explorer")


@st.cache_resource
def init_ee():
    ee.Initialize(project=EE_PROJECT)


init_ee()

AOI = ee.Geometry.Rectangle([76.55, 10.65, 76.90, 10.95], None, False)
S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
CLASS_NAMES = ["Dense Forest", "Agroforestry / TOF", "Cropland", "Built-up", "Water"]
CLASS_PALETTE = ["0b6623", "76b041", "e3c700", "d7191c", "2c7fb8"]


# ---------------------------------------------------------------------------
# 1. REBUILD THE PIPELINE (condensed -- see 01_landcover_classification.py
#    for full comments). Cached so it only runs once per session.
# ---------------------------------------------------------------------------
def mask_s2_sr(img):
    scl = img.select("SCL")
    good_mask = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7))
    return (
        img.updateMask(good_mask)
        .select(S2_BANDS)
        .divide(10000)
        .copyProperties(img, ["system:time_start"])
    )


def seasonal_composite(start, end):
    return (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(AOI)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
        .map(mask_s2_sr)
        .median()
        .clip(AOI)
    )


def add_indices(img, tag):
    ndvi = img.normalizedDifference(["B8", "B4"]).rename(f"NDVI_{tag}")
    evi = img.expression(
        "2.5*((NIR-RED)/(NIR+6*RED-7.5*BLUE+1))",
        {"NIR": img.select("B8"), "RED": img.select("B4"), "BLUE": img.select("B2")},
    ).rename(f"EVI_{tag}")
    ndwi = img.normalizedDifference(["B3", "B8"]).rename(f"NDWI_{tag}")
    ndbi = img.normalizedDifference(["B11", "B8"]).rename(f"NDBI_{tag}")
    return img.addBands([ndvi, evi, ndwi, ndbi])


@st.cache_resource
def build_classification():
    post_monsoon = add_indices(seasonal_composite("2023-10-01", "2023-12-31"), "pm")
    summer = add_indices(seasonal_composite("2024-02-01", "2024-04-30"), "sm")

    nir_int = post_monsoon.select("B8").multiply(255).toByte()
    texture = nir_int.glcmTexture(size=3).select("B8_contrast").rename("NIR_contrast_pm")
    ndvi_seasonal_diff = (
        post_monsoon.select("NDVI_pm")
        .subtract(summer.select("NDVI_sm"))
        .abs()
        .rename("NDVI_seasonal_diff")
    )

    feature_stack = (
        post_monsoon.select(S2_BANDS)
        .addBands(post_monsoon.select(["NDVI_pm", "EVI_pm", "NDWI_pm", "NDBI_pm"]))
        .addBands(summer.select(["NDVI_sm", "EVI_sm", "NDWI_sm", "NDBI_sm"]))
        .addBands(texture)
        .addBands(ndvi_seasonal_diff)
        .clip(AOI)
    )

    worldcover = ee.ImageCollection("ESA/WorldCover/v200").first().clip(AOI)
    tree_cover_pct = (
        ee.Image("UMD/hansen/global_forest_change_2023_v1_11")
        .select("treecover2000")
        .clip(AOI)
    )
    wc_tree = worldcover.eq(10)
    wc_cropland = worldcover.eq(40)
    wc_builtup = worldcover.eq(50)
    wc_water = worldcover.eq(80).Or(worldcover.eq(90))

    dense_forest = wc_tree.And(tree_cover_pct.gte(70))
    agroforestry_from_tree = wc_tree.And(tree_cover_pct.lt(70))
    agroforestry_from_crop = wc_cropland.And(tree_cover_pct.gte(10)).And(tree_cover_pct.lt(40))
    pure_cropland = wc_cropland.And(tree_cover_pct.lt(10))

    label_image = (
        ee.Image(0)
        .where(dense_forest, 1)
        .where(agroforestry_from_tree.Or(agroforestry_from_crop), 2)
        .where(pure_cropland, 3)
        .where(wc_builtup, 4)
        .where(wc_water, 5)
        .rename("class")
        .updateMask(
            ee.Image(0)
            .where(
                dense_forest.Or(agroforestry_from_tree)
                .Or(agroforestry_from_crop)
                .Or(pure_cropland)
                .Or(wc_builtup)
                .Or(wc_water),
                1,
            )
            .selfMask()
        )
    )

    sample_pts = label_image.addBands(feature_stack).stratifiedSample(
        numPoints=150,
        classBand="class",
        region=AOI,
        scale=10,
        seed=42,
        geometries=True,
        classValues=[1, 2, 3, 4, 5],
        classPoints=[150, 150, 150, 100, 60],
    )
    with_random = sample_pts.randomColumn("rand", 42)
    train_set = with_random.filter(ee.Filter.lt("rand", 0.7))
    valid_set = with_random.filter(ee.Filter.gte("rand", 0.7))

    rf_classifier = ee.Classifier.smileRandomForest(
        numberOfTrees=200, minLeafPopulation=3, seed=42
    ).train(
        features=train_set,
        classProperty="class",
        inputProperties=feature_stack.bandNames(),
    )
    classified = feature_stack.classify(rf_classifier).rename("classification")

    confusion_matrix = valid_set.classify(rf_classifier).errorMatrix(
        "class", "classification"
    )
    overall_accuracy = confusion_matrix.accuracy().getInfo()
    kappa = confusion_matrix.kappa().getInfo()

    return classified, overall_accuracy, kappa


@st.cache_data
def compute_areas(_classified):
    pixel_area_km2 = ee.Image.pixelArea().divide(1e6)
    areas = {}
    for i, name in enumerate(CLASS_NAMES):
        class_val = i + 1
        area = (
            pixel_area_km2.updateMask(_classified.eq(class_val))
            .reduceRegion(
                reducer=ee.Reducer.sum(), geometry=AOI, scale=10, maxPixels=1e10
            )
            .get("area")
        )
        areas[name] = ee.Number(area).getInfo()
    return areas


# ---------------------------------------------------------------------------
# 2. LAYOUT
# ---------------------------------------------------------------------------
st.title("Palakkad Land-Cover Explorer")
st.caption("Random Forest classification of a Sentinel-2 (2-season) mosaic.")

with st.spinner("Running classification on Earth Engine (first load only)..."):
    classified, overall_accuracy, kappa = build_classification()
    areas = compute_areas(classified)

map_col, side_col = st.columns([3, 1])

with map_col:
    m = geemap.Map(center=[10.80, 76.72], zoom=11, height=650)
    m.add_basemap("SATELLITE")
    vis_params = {"min": 1, "max": 5, "palette": CLASS_PALETTE}
    m.addLayer(classified, vis_params, "Land Cover")
    m.to_streamlit(height=650)

with side_col:
    st.subheader("Legend")
    for name, color in zip(CLASS_NAMES, CLASS_PALETTE):
        st.markdown(
            f"<div style='display:flex;align-items:center;margin:4px 0;'>"
            f"<div style='width:16px;height:16px;background:#{color};"
            f"margin-right:8px;border:1px solid #999;'></div>{name}</div>",
            unsafe_allow_html=True,
        )

    st.subheader("Accuracy")
    st.metric("Overall Accuracy", f"{overall_accuracy:.3f}")
    st.metric("Kappa Coefficient", f"{kappa:.3f}")

    st.subheader("Class Area (sq km)")
    for name, area in areas.items():
        st.write(f"**{name}**: {area:.1f}")

st.caption(
    "Data: Sentinel-2 SR (post-monsoon + summer composites), "
    "ESA WorldCover v200 & Hansen GFC as bootstrap training reference."
)
