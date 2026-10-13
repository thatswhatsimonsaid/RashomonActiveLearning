import pandas as pd

from src.utils.preprocess_data import (
    frame_from_autompg,
    frame_from_concrete_strength,
    frame_from_cps,
    frame_from_pollution,
    frames_from_concrete_slump,
)


def test_concrete_slump_makes_three_responses():
    table = pd.DataFrame(
        [
            [1, 10, 1, 2, 3, 4, 5, 6, 23, 62, 34.5],
            [2, 11, 1, 2, 3, 4, 5, 6, 0, 20, 41.0],
        ]
    )
    frames = frames_from_concrete_slump(table)
    assert set(frames) == {"concrete-slump", "concrete-flow", "concrete-cs"}
    assert frames["concrete-slump"].shape == (2, 8)
    assert frames["concrete-flow"]["Y"].tolist() == [62.0, 20.0]
    assert frames["concrete-cs"]["Y"].tolist() == [34.5, 41.0]
    assert frames["concrete-slump"]["V0"].tolist() == [10.0, 11.0]


def test_autompg_drops_missing_horsepower_and_encodes_origin():
    rows = [
        [18.0, 8, 307.0, 130.0, 3504, 12.0, 70, 1, "chevrolet"],
        [16.0, 4, 120.0, 95.0, 2300, 15.0, 72, 2, "datsun"],
        [15.0, 8, 350.0, "?", 3693, 11.5, 70, 1, "buick"],
        [17.0, 4, 100.0, 90.0, 2200, 14.0, 71, 3, "toyota"],
    ]
    frame = frame_from_autompg(pd.DataFrame(rows))
    assert len(frame) == 3
    assert frame.shape[1] == 10
    assert frame["Y"].tolist() == [18.0, 16.0, 17.0]
    assert frame["V2"].tolist() == [130.0, 95.0, 90.0]
    assert frame[["V6", "V7", "V8"]].to_numpy().tolist() == [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]


def test_pollution_uses_the_first_column_as_the_response():
    table = pd.DataFrame([[3.5, 1, 2, 3, 4, 5, 6, 7], [4.5, 0, 0, 0, 0, 0, 0, 1]])
    frame = frame_from_pollution(table)
    assert frame.shape == (2, 8)
    assert frame["Y"].tolist() == [3.5, 4.5]
    assert frame["V0"].tolist() == [1.0, 0.0]


def test_cps_omits_race_and_one_hots_occupation_and_sector():
    base = [12, 0, 1, 9, 0, 8.5, 27, 3, 6, 1, 1]
    other_race = [12, 0, 1, 9, 0, 8.5, 27, 1, 6, 1, 1]
    changed_job = [12, 0, 1, 9, 0, 9.0, 27, 3, 1, 0, 0]
    frame = frame_from_cps(pd.DataFrame([base, other_race, changed_job]))
    assert frame.iloc[0].tolist() == frame.iloc[1].tolist()
    assert frame.shape[1] == 1 + 7 + 2 + 2
    assert frame["Y"].tolist()[2] == 9.0


def test_concrete_strength_holds_out_the_last_column():
    table = pd.DataFrame(
        [["cement", "slag", "ash", "water", "sp", "coarse", "fine", "age", "strength"],
         [540, 0, 0, 162, 2.5, 1040, 676, 28, 79.9]]
    )
    frame = frame_from_concrete_strength(table.iloc[1:])
    assert frame.shape == (1, 9)
    assert frame["Y"].tolist() == [79.9]
    assert frame["V7"].tolist() == [28.0]
