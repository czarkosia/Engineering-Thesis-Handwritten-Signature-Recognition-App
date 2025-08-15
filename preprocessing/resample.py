import numpy as np

#Menger curvature
def calculate_curvature(point_a: tuple[int, int], point_b: tuple[int, int], point_c: tuple[int, int]) -> float:
    dx_ba, dy_ba = point_b[0] - point_a[0], point_b[1] - point_a[1]
    dx_ca, dy_ca = point_c[0] - point_a[0], point_c[1] - point_a[1]
    dx_cb, dy_cb = point_c[0] - point_b[0], point_c[1] - point_b[1]

    area = 0.5 * np.abs(dx_ba * dy_ca - dx_ca * dy_ba)
    dist_ab = np.sqrt(dx_ba ** 2 + dy_ba ** 2)
    dist_ac = np.sqrt(dx_ca ** 2 + dy_ca ** 2)
    dist_bc = np.sqrt(dx_cb ** 2 + dy_cb ** 2)

    return 4 * area / (dist_ab * dist_ac * dist_bc)

def resample_all(dataset: list[np.ndarray], max_sequence_length: int):
    for j, signature in enumerate(dataset):
        to_delete = signature.shape[1] - max_sequence_length
        print("max_sequence_length:", max_sequence_length)
        print("signature.shape:", signature.shape)
        if to_delete <= 0:
            continue

        curvatures = []
        xy_coords = signature[0:2]
        for i in range(1, xy_coords.shape[1] - 2):
            p1 = (xy_coords[0][i - 1], xy_coords[1][i - 1])
            p2 = (xy_coords[0][i], xy_coords[1][i])
            p3 = (xy_coords[0][i + 1], xy_coords[1][i + 1])
            curvatures.append(calculate_curvature(p1, p2, p3))

        print(to_delete)
        for i in range(to_delete):
            idx = np.argmin(curvatures)
            print(idx, curvatures.pop(idx))
            signature = np.delete(signature, idx+1, 1)
            dataset[j] = signature

    return dataset