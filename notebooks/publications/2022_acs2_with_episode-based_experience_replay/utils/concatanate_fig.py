from PIL import Image
import os

OUTPUT_PATH = "FIGS"
BASE_PATH = os.path.join("RESULTS", "MAZE")
MAZE_ROW_1 = ["F1", "F2", "F3", "T2", "T3"]
MAZE_ROW_2 = ["4", "5", "7", "Z", "XYZ"]

METRICS = ["steps_explore_with_dqn", "knowledge", "optimal"]


def get_file_path(maze, metric):
    return os.path.join(BASE_PATH, f'Maze{maze}-v0', f'MAZE_{maze}_EXP_EXPLORE_1', f'{metric}.png')


def get_concat_h(im1, im2, im3, im4, im5):
    dst = Image.new('RGB', (im1.width + im2.width +
                            im3.width + im4.width + im5.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    dst.paste(im3, (im1.width + im2.width, 0))
    dst.paste(im4, (im1.width + im2.width + im3.width, 0))
    dst.paste(im5, (im1.width + im2.width + im3.width + im4.width, 0))
    return dst


def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


os.makedirs(OUTPUT_PATH, exist_ok=True)


def get_row(maze_row, metric):
    im1 = Image.open(get_file_path(maze_row[0], metric))
    im2 = Image.open(get_file_path(maze_row[1], metric))
    im3 = Image.open(get_file_path(maze_row[2], metric))
    im4 = Image.open(get_file_path(maze_row[3], metric))
    im5 = Image.open(get_file_path(maze_row[4], metric))

    return get_concat_h(im1, im2, im3, im4, im5)


for metric in METRICS:
    row_1 = get_row(MAZE_ROW_1, metric)
    row_2 = get_row(MAZE_ROW_2, metric)

    full_img = get_concat_v(row_1, row_2)

    metric_result_path = os.path.join(OUTPUT_PATH, f'{metric}.png')
    full_img.save(metric_result_path, quality=600)
