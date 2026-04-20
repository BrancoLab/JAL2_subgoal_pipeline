import numpy as np
import polars as pl


def find_shelter_exit_runs(video_df, min_speed_cm_s=5.0, min_distance_cm=50.0, cm_per_pixel=.1):  # set this if x/y are in pixels
    outside = video_df["OutofshelterIdx"].to_numpy().astype(bool)
    speed = video_df["speed"].to_numpy().astype(float)
    x = video_df["mouse_x_position"].to_numpy().astype(float) * cm_per_pixel
    y = video_df["mouse_y_position"].to_numpy().astype(float) * cm_per_pixel
    outside_runs = np.zeros_like(outside, dtype=bool)

    n = len(outside)

    # Frames where mouse transitions from inside -> outside
    exit_frames = np.where((~outside[:-1]) & (outside[1:]))[0] + 1

    for start in exit_frames:
        # Must leave shelter and be moving fast immediately
        # TODO: loosen criteria?
        if speed[start] < min_speed_cm_s:
            continue

        end = start
        while end + 1 < n:
            next_i = end + 1

            # Stop condition 1: speed drops below threshold
            if speed[next_i] < min_speed_cm_s:
                break

            # Stop condition 2: re-enters shelter
            if not outside[next_i]:
                break

            end = next_i

        # Compute path length (cm) over the uninterrupted run
        if end > start:
            dx = np.diff(x[start : end + 1])
            dy = np.diff(y[start : end + 1])
            distance_cm = np.sum(np.hypot(dx, dy))
        else:
            distance_cm = 0.0

        if distance_cm >= min_distance_cm:
            outside_runs[start : end + 1] = True

    return outside_runs

def find_bout_runs(
    video_df,
    run_speed_cm_s=5.0,
    min_distance_cm=20.0,
    cm_per_pixel=0.1,
    max_dip_frames=2,
    stop_speed_cm_s=1.0,
    min_stop_frames=4,
    max_homing_overlap_frames=1,   # throw out run if overlap is greater than this
    remove_shelter_outings=False,  # if True, it will remove runs that overlap with shelter outings as defined by find_shelter_exit_runs
):
    speed = video_df["speed"].to_numpy().astype(float)
    x_cm = video_df["mouse_x_position"].to_numpy().astype(float) * cm_per_pixel
    y_cm = video_df["mouse_y_position"].to_numpy().astype(float) * cm_per_pixel
    homing = video_df["homingPeriod"].to_numpy().astype(bool)
    shelter_outings = find_shelter_exit_runs(video_df, min_distance_cm = 20.0)

    n = len(speed)
    above = speed >= run_speed_cm_s

    if max_dip_frames > 0:
        i = 0
        while i < n:
            if above[i]:
                i += 1
                continue
            j = i
            while j < n and not above[j]:
                j += 1
            gap_len = j - i
            has_left_run = i > 0 and above[i - 1]
            has_right_run = j < n and above[j]
            if has_left_run and has_right_run and gap_len <= max_dip_frames:
                above[i:j] = True
            i = j

    run_mask = np.zeros(n, dtype=bool)
    starts = np.where((~above[:-1]) & (above[1:]))[0] + 1
    ends = np.where((above[:-1]) & (~above[1:]))[0]
    if above[0]:
        starts = np.r_[0, starts]
    if above[-1]:
        ends = np.r_[ends, n - 1]

    for s, e in zip(starts, ends):
        seg_speed = speed[s:e + 1]
        low = seg_speed <= stop_speed_cm_s

        split_points = []
        if min_stop_frames > 0:
            k = 0
            while k < len(low):
                if not low[k]:
                    k += 1
                    continue
                m = k
                while m < len(low) and low[m]:
                    m += 1
                if (m - k) >= min_stop_frames:
                    split_points.append((k, m - 1))
                k = m

        if not split_points:
            local_intervals = [(0, len(seg_speed) - 1)]
        else:
            local_intervals = []
            cur = 0
            for a, b in split_points:
                if a - 1 >= cur:
                    local_intervals.append((cur, a - 1))
                cur = b + 1
            if cur <= len(seg_speed) - 1:
                local_intervals.append((cur, len(seg_speed) - 1))

        for ls, le in local_intervals:
            gs = s + ls
            ge = s + le
            if ge <= gs:
                continue

            dx = np.diff(x_cm[gs:ge + 1])
            dy = np.diff(y_cm[gs:ge + 1])
            dist_cm = np.sum(np.hypot(dx, dy))
            if dist_cm < min_distance_cm:
                continue

            overlap_frames = np.sum(homing[gs:ge + 1])
            if overlap_frames > max_homing_overlap_frames:
                continue
            if remove_shelter_outings:
                shelter_overlap_frames = np.sum(shelter_outings[gs:ge + 1])
                if shelter_overlap_frames > 0:
                    continue

            run_mask[gs:ge + 1] = True

    return run_mask