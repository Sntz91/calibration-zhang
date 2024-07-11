# mainly copied from https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
import cv2
import numpy as np
import glob
import json
from typing import Tuple
import argparse

def calculate_camera_calibration_matrix(
        folderpath: str,
        image_format: str,
        chessboard_format: tuple,
        draw_corners: bool=False,
):
    """
        Calculates a camera calibration Matrix from checkered board Images
        Input: Path to dir with Calibration Images, Chessboard Corners Format (b x h)
        Output: Intrinsic Camera Parameters Matrix
    """
    termination_criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001
    )

    files = f"{folderpath}/*{image_format}"
    image_fnames = glob.glob(files)
    assert len(image_fnames) > 0, "No image found."

    obj_points = len(image_fnames)*[
        np.array(
            np.meshgrid(range(0, 8+1), range(0, 5+1), 0, indexing='ij'), 
            np.float32
        ).T.reshape(-1, 3)
    ]
    img_points = []

    for fname in image_fnames:
        print("Calibrating File:" , fname)
        img = cv2.imread(fname)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(
            img_gray, chessboard_format, None
        )
        assert ret, "No Chessboard Corners found."
        corners2 = cv2.cornerSubPix( # RENAME! WHAT ARE CORNERS2 DOIGN
            img_gray, corners, (11, 11), (-1, -1), termination_criteria
        )
        img_points.append(corners2)

        if draw_corners:
            cv2.drawChessboardCorners(img, chessboard_format, corners2, ret)
            cv2.imshow(fname, img)
            cv2.waitKey()
            cv2.destroyAllWindows()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, img_gray.shape[::-1], None, None
    )
    return mtx, dist, rvecs, tvecs


def dump_camera_information_to_json(folderpath, filename, cam_dict):
    with open(folderpath + "/" + filename, 'w') as fp:
        json.dump(cam_dict, fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_name", help="path name", required=True, type=str)
    parser.add_argument("--image_format", help="images of jpg / png etc.", required=True, type=str)
    parser.add_argument("--chessboard_format", help="chessboard size", required=True, nargs="+", type=int)
    parser.add_argument("--show_boards", help="show boards", default=False)
    # assert len(sys.argv) == 2, "Need name of folder as argument. Nothing else."
    args = parser.parse_args()
    print(args.chessboard_format)

    mat, dist, rot, trans = calculate_camera_calibration_matrix(
        f"./{args.path_name}/calibration_images", 
        f".{args.image_format}", 
        args.chessboard_format, 
        draw_corners=args.show_boards
    )

    # np.array to lists for json
    cam_dict = {
        "mtx": mat.tolist(), 
        "dist": dist.tolist(), 
        "rot": [r.tolist() for r in rot], 
        "trans": [t.tolist() for t in trans]
    }

    img_to_undist = cv2.imread(f"./{args.path_name}/reference_image.jpg")
    img_undist = cv2.undistort(img_to_undist, mat, dist)
    cv2.imshow('undistorted', img_undist)
    cv2.imshow('original', img_to_undist)
    cv2.waitKey()
   
    dump_camera_information_to_json(f"./{args.path_name}", "calibration.json", cam_dict)
