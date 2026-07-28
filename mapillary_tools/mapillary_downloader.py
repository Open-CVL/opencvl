"""
This code can be used to download Mapillary images in a bounding box specified
by the user in latitude/longitude format.

For every image, detailed labels are also retrieved from Mapillary, including
the raw-reported poses, OpenSfM-corrected poses, and other metadata. Retrying
and tiling are provided to navigate the Mapillary API access limits.
"""

# %%
import io
import json
import os
import time

import cv2
import mapillary.interface as mly
import numpy as np
import requests
from PIL import ExifTags, Image


# %%
"""
CONFIGURATION.
Please change accordingly.
"""

# Get your personal token from Mapillary.com by creating a user account.
MAPILLARY_TOKEN = "MLY|XXXXXXXXX"

# Local paths for downloaded Mapillary images and their labels.
outpath_groundimgs = "../testregiondownloads/mapillary_images"
outpath_labels_json = "../testregiondownloads/labels"

mly.set_access_token(MAPILLARY_TOKEN)

# Large-area bounding box (lon_min, lat_min, lon_max, lat_max).
BIG_BBOX = [4.3171347, 52.0137663, 4.5227971, 53.0983924]

# Tile a large area into smaller requests due to the Mapillary download limit.
TILE_SIZE = 0.02
LIMIT = 100


# %%
def tile_bbox(bbox, tile_size):
    min_lon, min_lat, max_lon, max_lat = bbox
    tiles = []

    lon = min_lon
    while lon < max_lon:
        lat = min_lat
        while lat < max_lat:
            tiles.append(
                [
                    lon,
                    lat,
                    min(lon + tile_size, max_lon),
                    min(lat + tile_size, max_lat),
                ]
            )
            lat += tile_size
        lon += tile_size

    return tiles


def fetch_images_for_tile(bbox, limit=100, max_retries=5):
    images_data = []
    images_ids = []

    url = "https://graph.mapillary.com/images"
    params = {
        "access_token": MAPILLARY_TOKEN,
        "bbox": ",".join(map(str, bbox)),
        "fields": "id,geometry",
        "limit": limit,
    }

    while True:
        attempt = 0
        while attempt < max_retries:
            try:
                response = requests.get(url, params=params, timeout=120)
                response.raise_for_status()
                data = response.json()
                break
            except requests.exceptions.HTTPError as error:
                if response.status_code == 500:
                    attempt += 1
                    print(
                        f"500 Server Error for bbox {bbox}, "
                        f"retry {attempt}/{max_retries}"
                    )
                    time.sleep(1)
                else:
                    raise error
            except requests.exceptions.RequestException as error:
                attempt += 1
                print(
                    f"Request error: {error}, "
                    f"retry {attempt}/{max_retries}"
                )
                time.sleep(1)
        else:
            print(
                f"Skipping bbox {bbox} after "
                f"{max_retries} failed attempts"
            )
            return images_ids, images_data

        for image in data.get("data", []):
            image_id = image["id"]

            try:
                detailed_data = json.loads(mly.image_from_key(image_id))
                detailed_data["features"]["geometry"] = image["geometry"]
                images_ids.append(image_id)
                images_data.append(detailed_data)
            except Exception as error:
                print(
                    f"Failed to fetch detailed data for image "
                    f"{image_id}: {error}"
                )

        paging = data.get("paging", {})
        if "next" not in paging:
            break

        url = paging["next"]
        params = None

    return images_ids, images_data


def download_mapillaryimage(url):
    if not url:
        return None

    response = requests.get(url, stream=True, timeout=20)
    image_stream = io.BytesIO(response.content)
    return Image.open(image_stream)


def undistort_image(
    img_mapil,
    camera_matrix_mapil,
    distortion_mapil,
    image_dimensions,
    camera_type,
):
    """
    Undistort an image to match the undistorted images supplied by Open-CVL.
    """

    if camera_type == "perspective":
        print("using perspective model")

        new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
            camera_matrix_mapil,
            distortion_mapil,
            image_dimensions,
            0,
            image_dimensions,
        )
        map_x, map_y = cv2.initUndistortRectifyMap(
            new_camera_matrix,
            distortion_mapil,
            None,
            new_camera_matrix,
            image_dimensions,
            cv2.CV_32FC2,
        )
        return cv2.remap(
            img_mapil,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
        )

    if camera_type == "fisheye":
        print("using fisheye model")

        new_camera_matrix = (
            cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                camera_matrix_mapil,
                distortion_mapil,
                image_dimensions,
                np.eye(3),
                balance=0.0,
            )
        )
        map_1, map_2 = cv2.fisheye.initUndistortRectifyMap(
            camera_matrix_mapil,
            distortion_mapil,
            np.eye(3),
            new_camera_matrix,
            image_dimensions,
            cv2.CV_16SC2,
        )
        return cv2.remap(
            img_mapil,
            map_1,
            map_2,
            interpolation=cv2.INTER_LINEAR,
        )

    raise ValueError(f"Unsupported camera type: {camera_type}")


def apply_exif_orientation(img, orientation):
    """
    Apply EXIF orientation to a PIL image.

    Some Mapillary images are stored with an orientation that needs to be
    applied before converting the image to a NumPy array.
    """

    if orientation == 1 or orientation is None:
        return img
    if orientation == 2:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    if orientation == 3:
        return img.rotate(180, expand=True)
    if orientation == 4:
        return img.transpose(Image.FLIP_TOP_BOTTOM)
    if orientation == 5:
        return img.transpose(Image.FLIP_LEFT_RIGHT).rotate(270, expand=True)
    if orientation == 6:
        return img.rotate(270, expand=True)
    if orientation == 7:
        return img.transpose(Image.FLIP_LEFT_RIGHT).rotate(90, expand=True)
    if orientation == 8:
        return img.rotate(90, expand=True)
    return img


# %%
def download_and_store_imageswithlabels(
    img_id,
    detaileddata,
    groundimgs_dir_path,
    labels_json_path,
):
    try:
        properties = detaileddata["features"]["properties"]
        url = properties["thumb_original_url"]
        seq_name = properties["sequence"]
        mapil_reported_width = properties["width"]
        mapil_reported_height = properties["height"]
        lonlat_mapil_raw = detaileddata["features"]["geometry"]["coordinates"]
        heading_mapil_raw = properties["compass_angle"]
        creator_id = properties["creator_id"]
        camera_params_raw = properties["camera_parameters"]
        lonlat_mapil_corr = properties["computed_geometry"]["coordinates"]
        heading_mapil_corr = properties["computed_compass_angle"]
        camera_type = properties["camera_type"]

        url = json.loads(f'"{url}"')
        if not url:
            return

        img_mapil = download_mapillaryimage(url)
        exif = img_mapil._getexif()
        if exif is not None:
            for tag_id, value in exif.items():
                tag = ExifTags.TAGS.get(tag_id, tag_id)
                if tag == "Orientation":
                    img_mapil = apply_exif_orientation(img_mapil, value)

        img_mapil = np.array(img_mapil)

        distortion_mapil = np.zeros(4, np.float32)
        focal_length = camera_params_raw[0] * max(
            mapil_reported_width,
            mapil_reported_height,
        )
        distortion_mapil[0] = camera_params_raw[1]
        distortion_mapil[1] = camera_params_raw[2]
        center_x = int(mapil_reported_width / 2.0)
        center_y = int(mapil_reported_height / 2.0)
        camera_matrix_mapil = np.array(
            [
                [focal_length, 0, center_x],
                [0, focal_length, center_y],
                [0, 0, 1],
            ],
            np.float32,
        )

        image_dimensions = (mapil_reported_width, mapil_reported_height)
        img_mapil_undist = undistort_image(
            img_mapil,
            camera_matrix_mapil,
            distortion_mapil,
            image_dimensions,
            camera_type,
        )

        cv2.imwrite(
            os.path.join(groundimgs_dir_path, f"{img_id}.png"),
            cv2.cvtColor(img_mapil_undist, cv2.COLOR_RGB2BGR),
        )

        relevant_labels = {
            "latlon_mapilraw": [
                lonlat_mapil_raw[1],
                lonlat_mapil_raw[0],
            ],
            "latlon_mapilopensfm": [
                lonlat_mapil_corr[1],
                lonlat_mapil_corr[0],
            ],
            "heading_mapilraw": heading_mapil_raw,
            "heading_mapilopensfm": heading_mapil_corr,
            "mapil_intrinsics_3x3": camera_matrix_mapil.tolist(),
            "camera_type": camera_type,
            "mapil_image_id": img_id,
            "mapil_sequence_id": seq_name,
            # Useful for attribution under the Mapillary CC-BY-SA license.
            "mapil_creator_id": creator_id,
        }

        with open(
            os.path.join(labels_json_path, f"{img_id}.json"),
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(relevant_labels, file)
        print(
            "Saved Mapillary data corresponding to "
            f"Mapillary image ID {img_id}"
        )

    except Exception as error:
        print(
            "Failed to retrieve and store data for "
            f"Mapillary image ID {img_id}: {error}"
        )
