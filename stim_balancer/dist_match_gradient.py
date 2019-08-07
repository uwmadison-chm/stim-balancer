#!/usr/bin/env python

from dataclasses import dataclass
from typing import List, Any, Mapping, Set, Tuple

import statsmodels.api as sm
import pandas as pd

import logging
logger = logging.getLogger(__name__)


"""
A gradient descent-based method for matching stimulis lists based on various
parameters.

Takes a dataframe and grouping/testing information and tries to create groups
with similar means, by doing the following:

1: Draw N random starting samples from each group. Smaller N means easier
   matching, larger N means more randomness.
   Example: pos, neu, neg groups, starting with 10 samples per group, target
   size is 50 samples,
2: Iterate over each group G that needs more points; we're going to add a point
   to this group.
   Example: We're going to add a point to 'pos':
    a: For each test T for the group, compute the difference for this test
       between the other groups.
       Example: pos-neu on arousal, 0.7, pos-neg on arousal, 0.02
    b: Take the smallest p-value in this list of tests.
       Example: pos-neg, arousal
    c: Find the point in the pool that will bring our mean closest to the mean
       of the other group, add this to our selected points.
    d: Remove that point from the pool
3: Goto 2 until no groups need points
"""


@dataclass
class MatchDef:
    column: str
    weight: float = 1.0


@dataclass
class GroupDef:
    """
    Information about a grouping operation. Holds a pool of points,
    a list of which points are used, and how many points we need
    """
    group_value: Any
    target_size: int
    matchdefs: List[MatchDef]
    point_pool: pd.DataFrame = None
    used_point_mask: pd.Series = None

    def pick_seed_points(self, n, random_state=0):
        self.used_point_mask[:] = False
        sample = self.used_point_mask.sample(n=n, random_state=random_state)
        self.used_point_mask[sample.index] = True
        logger.debug(f"Picked {self.used_point_mask.sum()} points")

    @property
    def columns(self):
        return [md.column for md in self.matchdefs]

    @property
    def weights(self):
        return [md.weight for md in self.matchdefs]

    @property
    def cols_to_zero(self):
        return [md.column for md in self.matchdefs if md.compare_to_zero]

    @property
    def used_points(self):
        return self.point_pool.loc[self.used_point_mask]

    @property
    def available_points(self):
        return self.point_pool.loc[~self.used_point_mask]

    @property
    def used_point_count(self):
        return self.used_point_mask.sum()

    @property
    def needed_point_count(self):
        return self.target_size - self.used_point_count

    @property
    def needs_points(self):
        return self.needed_point_count > 0

    @property
    def is_full(self):
        return not self.needs_points

    @property
    def debug_str(self):
        return f'GroupDef(group_value={repr(self.group_value)}, target_size={self.target_size}, used_point_count={self.used_point_count}, needed_point_count={self.needed_point_count})'


class GroupDefSet:
    groupdefs: List[GroupDef]
    group_column: str

    def __init__(self, groupdefs, group_column):
        self.groupdefs = groupdefs
        self.group_column = group_column
        self.mapping = {g.group_value: g for g in groupdefs}
        self.group_values = set(self.mapping.keys())

    def init_working_spaces(self, data_frame: pd.DataFrame):
        for groupdef in self.groupdefs:
            logger.debug(
                f"Searching for rows where {self.group_column} == {groupdef.group_value}")
            point_pool = data_frame.loc[
                data_frame[self.group_column] == groupdef.group_value
            ].copy()
            logger.debug(f"Found shape: {point_pool.shape}")
            used_point_mask = pd.Series(False, index=point_pool.index)
            groupdef.point_pool = point_pool
            groupdef.used_point_mask = used_point_mask

    def __iter__(self):
        return iter(self.groupdefs)

    def fill_all_points(self):
        for i in range(self.needed_point_count):
            self.fill_point_for_all_groupdefs()

    def fill_point_for_all_groupdefs(self):
        for groupdef in self.groupdefs:
            self.fill_best_point(groupdef)

    def fill_best_point(self, groupdef: GroupDef):
        logger.debug(f'Filling point for {groupdef.debug_str}')
        if groupdef.is_full:
            logger.debug(f'Already full, returning...')
            return
        column, match_group_val = self._get_match_target(groupdef)
        match_groupdef = self.mapping[match_group_val]
        target_mean = match_groupdef.used_points[column].mean()
        best_index = most_harmonious_index_for_group(
            groupdef, column, target_mean)
        groupdef.used_point_mask[best_index] = True
        logger.debug(
            f'Filled point {best_index}, need {groupdef.needed_point_count} more points'
        )

    def other_group_values(self, group_value) -> Set[str]:
        return self.group_values - {group_value}

    def all_pvalues(self) -> pd.Series:
        pval_series_list = []
        for groupdef in self.groupdefs:
            res = self._pvalues_for(groupdef)
            res.index = [
                (
                    column_name,
                    f'{groupdef.group_value}-{contrast_group_value}'
                )
                for column_name, contrast_group_value
                in res.index
            ]
            pval_series_list.append(res)
        return pd.concat(pval_series_list)

    def _build_design(
            self,
            group_value: str,
            dv_column_name: str) -> Tuple[pd.Series, pd.DataFrame]:
        other_values = list(self.other_group_values(group_value))
        base_col_names = ['Intercept'] + other_values
        col_names = [(dv_column_name, c) for c in base_col_names]

        dm = pd.DataFrame(
            index=self.df.index,
            columns=col_names,
            data=0.0)
        dm[(dv_column_name, 'Intercept')] = 1.0
        for group_val in other_values:
            group_col_name = (dv_column_name, group_val)
            dm.loc[
                self.df[self.group_column] == group_val, group_col_name] = 1.0
        return (self.df[dv_column_name], dm)

    def _get_match_target(self, groupdef: GroupDef):
        pvals = self._pvalues_for(groupdef)
        min_pval = pvals.min()
        min_idx = pvals.idxmin()
        logger.debug(f'Min pval: {min_pval} for {min_idx}')
        return min_idx

    def _pvalues_for(self, groupdef: GroupDef) -> pd.Series:
        all_pvals = [
            self._pvalues_for_matchdef(groupdef.group_value, md)
            for md in groupdef.matchdefs]
        return pd.concat(all_pvals)

    def _pvalues_for_matchdef(
            self, group_value: str, matchdef: MatchDef) -> pd.Series:
        y, X = self._build_design(group_value, matchdef.column)
        model = sm.OLS(y, X, missing='drop')
        result = model.fit()
        # Drop the intecept term, we don't care about it and it'll always be
        # basically zero
        # Also fill in 1.0 for any nan values (will happen if one group
        # is nan for our matchdef) -- those aren't the values we're looking for
        pvals = result.pvalues.iloc[1:].fillna(1.0)
        return pvals

    @property
    def df(self) -> pd.DataFrame:
        return pd.concat(g.used_points for g in self.groupdefs)

    @property
    def needed_point_count(self):
        return max([gd.needed_point_count for gd in self.groupdefs])


def fill_lists(groupdefs: GroupDefSet):
    logger.debug(f"Need to pick {groupdefs.needed_point_count} points")
    for i in range(groupdefs.needed_point_count):
        logger.debug(f"Running iteration {i}")
        groupdefs.fill_point_for_all_groupdefs()


def most_harmonious_index_for_group(
        groupdef: GroupDef, column_name: str, target_mean: float) -> Any:
    target_value = value_for_target_mean(
        groupdef.used_points[column_name],
        target_mean
    )
    available_values = groupdef.available_points[column_name]
    distances = (available_values - target_value).abs()
    min_dist = distances.min()
    min_idx = distances.idxmin()
    logger.debug(f'Closest value: {min_dist} at index {min_idx}')
    return distances.idxmin()


def most_harmonious_index(
        used_vec: pd.Series, avail_vec: pd.Series, target_mean: float) -> Any:
    target_value = value_for_target_mean(used_vec, target_mean)
    distances = (avail_vec - target_value).abs()
    min_dist = distances.min()
    min_idx = distances.idxmin()
    best_val = avail_vec[min_idx]
    logger.debug(f"Trying to shift mean to {target_mean}, need {target_value}. Found {best_val} at {min_idx}, dist {min_dist}")
    return min_idx


def value_for_target_mean(vec: pd.Series, target_mean: float) -> float:
    next_n = len(vec) + 1
    cur_sum = vec.sum()
    return (next_n * target_mean) - cur_sum
