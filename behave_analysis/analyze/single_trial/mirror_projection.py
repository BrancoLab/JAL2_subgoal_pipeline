# This class contains a quick attempt at making a mirrow projection of the data. It is not used in the final analysis
# but it kept here for reference. I didn't check whether it failed biologically or because of a bug in the code.
# TODO: revist if index is valid to doube check if this is a valid approach

class DataProjector:
    """Projects data from one half of the arena to the other half to create a mirror image of the data
    overlaying the x axis. This is useful if the neural data is symmetrical and the mouse does not
    distinguish between the two sides of the arena.

    Returns:
        -- (pl.Dataframe) A dataframe with the projected data
    """

    def __init__(self, video_df):
        og_video_df = video_df.to_pandas().copy()
        video_df = video_df.to_pandas().copy()
        self._len_of_df = len(video_df)  # For len unit tests
        test = self.project_data_into_a_semi_circle(video_df)  # For integration tests

        self.resulting_df = pl.DataFrame(self.project_data_into_a_semi_circle(video_df))

        # Intergration test
        self._og_hsa = og_video_df["hsa"].to_numpy()  # For unit tests
        self._og_h_bar_north_a = og_video_df["h_bar_north_a"].to_numpy()  # For unit tests
        self._og_h_bar_south_a = og_video_df["h_bar_south_a"].to_numpy()  # For unit tests
        assert np.array_equal(test["hsa"], self._og_hsa), "The original hsa values have changed"
        assert np.array_equal(test["h_bar_south_a"], self._og_h_bar_south_a), "The original h_bar_south_a values have changed"
        assert np.array_equal(test["h_bar_north_a"], self._og_h_bar_north_a), "The original h_bar_north_a values have changed"

    def project_data_into_a_semi_circle(self, video_df) -> pd.DataFrame:
        """Project one half of the tracking data onto a semi circle creating a mirror along the centre x axis"""
        lhs_indexes = self.return_indexs_of_frames_on_the_left_hand_side_of_the_x_axis(video_df)
        video_df["hsa_mirror"] = self.invert_hsa_angle_to_mirror_image(lhs_indexes, video_df)
        assert len(video_df) == self._len_of_df, "The length of the dataframe has changed"

        # Create a mirror subgoal
        video_df["mirror_subgoal"] = self.project_south_subgoal_to_north_subgoal(video_df, lhs_indexes)
        assert len(video_df) == self._len_of_df, "The length of the dataframe has changed"

        return video_df

    def return_indexs_of_frames_on_the_left_hand_side_of_the_x_axis(self, video_df: pd.DataFrame, center_x=512) -> np.ndarray:
        """Return the indexes of the frames where the mouse is on the left hand side of the x axis
        The x coordinates range from 0 to 1024 with the shelter at the top"""
        indexes = video_df[np.asarray(video_df["mouse_x_position"] < center_x)].index.to_numpy()
        return indexes

    def invert_hsa_angle_to_mirror_image(self, inversion_indices: pd.Series, video_df: np.ndarray) -> np.ndarray:
        """Invert the angle to the mirror image of the angle if the mouse is on the left hand side of the x axis"""
        hsa = video_df["hsa"].to_numpy().copy()
        hsa[inversion_indices] = -hsa[inversion_indices]
        return hsa

    def project_south_subgoal_to_north_subgoal(self, video_df: pd.DataFrame, lhs_indices: np.ndarray) -> np.ndarray:
        # If the mouse is on the lhs then take the south subgoal, invert it and replace that indices in the north subgoal array
        north_subgoal = video_df["h_bar_north_a"].to_numpy().copy()
        south_subgoal = video_df["h_bar_south_a"].to_numpy().copy()
        assert len(north_subgoal) == len(south_subgoal), "The north and south subgoals are not the same length"

        north_subgoal[lhs_indices] = -south_subgoal[lhs_indices]
        mirror_subgoal = north_subgoal
        return mirror_subgoal

