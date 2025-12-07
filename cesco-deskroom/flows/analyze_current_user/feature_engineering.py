import json

import pandas as pd


def extract_work_features(df):
    """Extract work-related features from BIDB data - includes CSI features"""
    # Define expected columns for empty DataFrame case
    expected_columns = [
        "고객코드",
        "work_count",
        "unique_work_types",
        "confirmed_work",
        "cancelled_work",
        "work_span_days",
        "cancellation_rate",
        "confirmation_rate",
        "avg_services_per_work",
        "work_last_30d",
        "confirmed_30d",
        "cancelled_30d",
        "services_30d",
        "work_types_30d",
        "cancellation_rate_30d",
        "confirmation_rate_30d",
        "work_last_60d",
        "confirmed_60d",
        "cancelled_60d",
        "services_60d",
        "work_types_60d",
        "cancellation_rate_60d",
        "confirmation_rate_60d",
        "work_last_90d",
        "confirmed_90d",
        "cancelled_90d",
        "services_90d",
        "work_types_90d",
        "cancellation_rate_90d",
        "confirmation_rate_90d",
        "avg_csi_score",
        "csi_score_30d",
        "csi_score_60d",
        "csi_score_90d",
        "csi_survey_count",
        "csi_survey_count_30d",
        "csi_survey_count_60d",
        "csi_survey_count_90d",
    ]

    # Handle empty DataFrame case
    if df.empty:
        return pd.DataFrame(columns=expected_columns)

    features = []
    for idx, row in df.iterrows():
        history = row["작업이력"]
        num_work = len(history)

        work_types = [rec.get("작업유형") for rec in history if rec.get("작업유형")]
        unique_work_types = len(set(work_types))

        confirmed = sum([1 for rec in history if rec.get("확정여부") == True])
        cancelled = sum([1 for rec in history if rec.get("취소여부") == True])

        dates = []
        confirmed_dates = []
        cancelled_dates = []
        service_detail_dates = []
        csi_scores = []
        csi_dates = []

        for rec in history:
            date_str = rec.get("작업일자")
            if date_str:
                date = pd.to_datetime(date_str)
                dates.append(date)

                if rec.get("확정여부") == True:
                    confirmed_dates.append(date)
                if rec.get("취소여부") == True:
                    cancelled_dates.append(date)

                # Track dates when service details were present
                service_count = len(rec.get("서비스내역", []))
                for _ in range(service_count):
                    service_detail_dates.append(date)

                # Extract CSI scores
                satisfaction_data = rec.get("서비스_만족도", [])
                if satisfaction_data:
                    for satisfaction in satisfaction_data:
                        csi_score = satisfaction.get("평균_CSI_점수")
                        if csi_score is not None:
                            try:
                                csi_scores.append(float(csi_score))
                                csi_dates.append(date)
                            except (ValueError, TypeError):
                                pass

        if dates:
            last_date = max(dates)
            first_date = min(dates)
            work_span = (last_date - first_date).days

            # Work counts by time window
            work_30d = sum([1 for d in dates if (last_date - d).days <= 30])
            work_60d = sum([1 for d in dates if (last_date - d).days <= 60])
            work_90d = sum([1 for d in dates if (last_date - d).days <= 90])

            # Confirmed work by time window
            confirmed_30d = sum(
                [1 for d in confirmed_dates if (last_date - d).days <= 30]
            )
            confirmed_60d = sum(
                [1 for d in confirmed_dates if (last_date - d).days <= 60]
            )
            confirmed_90d = sum(
                [1 for d in confirmed_dates if (last_date - d).days <= 90]
            )

            # Cancelled work by time window
            cancelled_30d = sum(
                [1 for d in cancelled_dates if (last_date - d).days <= 30]
            )
            cancelled_60d = sum(
                [1 for d in cancelled_dates if (last_date - d).days <= 60]
            )
            cancelled_90d = sum(
                [1 for d in cancelled_dates if (last_date - d).days <= 90]
            )

            # Service details by time window (KEPT - time-windowed)
            services_30d = sum(
                [1 for d in service_detail_dates if (last_date - d).days <= 30]
            )
            services_60d = sum(
                [1 for d in service_detail_dates if (last_date - d).days <= 60]
            )
            services_90d = sum(
                [1 for d in service_detail_dates if (last_date - d).days <= 90]
            )

            # Work type diversity by time window
            work_types_30d = len(
                set(
                    [
                        rec.get("작업유형")
                        for rec in history
                        if rec.get("작업일자")
                        and (last_date - pd.to_datetime(rec.get("작업일자"))).days <= 30
                        and rec.get("작업유형")
                    ]
                )
            )
            work_types_60d = len(
                set(
                    [
                        rec.get("작업유형")
                        for rec in history
                        if rec.get("작업일자")
                        and (last_date - pd.to_datetime(rec.get("작업일자"))).days <= 60
                        and rec.get("작업유형")
                    ]
                )
            )
            work_types_90d = len(
                set(
                    [
                        rec.get("작업유형")
                        for rec in history
                        if rec.get("작업일자")
                        and (last_date - pd.to_datetime(rec.get("작업일자"))).days <= 90
                        and rec.get("작업유형")
                    ]
                )
            )

            # CSI calculations by time window
            if csi_scores and csi_dates:
                csi_scores_30d = [
                    score
                    for score, date in zip(csi_scores, csi_dates, strict=False)
                    if (last_date - date).days <= 30
                ]
                csi_scores_60d = [
                    score
                    for score, date in zip(csi_scores, csi_dates, strict=False)
                    if (last_date - date).days <= 60
                ]
                csi_scores_90d = [
                    score
                    for score, date in zip(csi_scores, csi_dates, strict=False)
                    if (last_date - date).days <= 90
                ]

                avg_csi_score = sum(csi_scores) / len(csi_scores)
                csi_score_30d = (
                    sum(csi_scores_30d) / len(csi_scores_30d) if csi_scores_30d else 0
                )
                csi_score_60d = (
                    sum(csi_scores_60d) / len(csi_scores_60d) if csi_scores_60d else 0
                )
                csi_score_90d = (
                    sum(csi_scores_90d) / len(csi_scores_90d) if csi_scores_90d else 0
                )

                csi_survey_count = len(csi_scores)
                csi_survey_count_30d = len(csi_scores_30d)
                csi_survey_count_60d = len(csi_scores_60d)
                csi_survey_count_90d = len(csi_scores_90d)
            else:
                avg_csi_score = 0
                csi_score_30d = csi_score_60d = csi_score_90d = 0
                csi_survey_count = csi_survey_count_30d = csi_survey_count_60d = (
                    csi_survey_count_90d
                ) = 0
        else:
            work_span = 0
            work_30d = work_60d = work_90d = 0
            confirmed_30d = confirmed_60d = confirmed_90d = 0
            cancelled_30d = cancelled_60d = cancelled_90d = 0
            services_30d = services_60d = services_90d = 0
            work_types_30d = work_types_60d = work_types_90d = 0
            # CSI defaults for no work history
            avg_csi_score = 0
            csi_score_30d = csi_score_60d = csi_score_90d = 0
            csi_survey_count = csi_survey_count_30d = csi_survey_count_60d = (
                csi_survey_count_90d
            ) = 0

        # Calculate total service details (for avg_services_per_work only)
        total_service_details = sum([len(rec.get("서비스내역", [])) for rec in history])

        features.append(
            {
                "고객코드": row["고객코드"],
                # Overall metrics
                "work_count": num_work,
                "unique_work_types": unique_work_types,
                "confirmed_work": confirmed,
                "cancelled_work": cancelled,
                "work_span_days": work_span,  # Will be removed from training later
                "cancellation_rate": cancelled / num_work if num_work > 0 else 0,
                "confirmation_rate": confirmed / num_work if num_work > 0 else 0,
                "avg_services_per_work": total_service_details / num_work
                if num_work > 0
                else 0,
                # 30-day window
                "work_last_30d": work_30d,
                "confirmed_30d": confirmed_30d,
                "cancelled_30d": cancelled_30d,
                "services_30d": services_30d,
                "work_types_30d": work_types_30d,
                "cancellation_rate_30d": cancelled_30d / work_30d
                if work_30d > 0
                else 0,
                "confirmation_rate_30d": confirmed_30d / work_30d
                if work_30d > 0
                else 0,
                # 60-day window
                "work_last_60d": work_60d,
                "confirmed_60d": confirmed_60d,
                "cancelled_60d": cancelled_60d,
                "services_60d": services_60d,
                "work_types_60d": work_types_60d,
                "cancellation_rate_60d": cancelled_60d / work_60d
                if work_60d > 0
                else 0,
                "confirmation_rate_60d": confirmed_60d / work_60d
                if work_60d > 0
                else 0,
                # 90-day window
                "work_last_90d": work_90d,
                "confirmed_90d": confirmed_90d,
                "cancelled_90d": cancelled_90d,
                "services_90d": services_90d,
                "work_types_90d": work_types_90d,
                "cancellation_rate_90d": cancelled_90d / work_90d
                if work_90d > 0
                else 0,
                "confirmation_rate_90d": confirmed_90d / work_90d
                if work_90d > 0
                else 0,
                # CSI features
                "avg_csi_score": avg_csi_score,
                "csi_score_30d": csi_score_30d,
                "csi_score_60d": csi_score_60d,
                "csi_score_90d": csi_score_90d,
                "csi_survey_count": csi_survey_count,
                "csi_survey_count_30d": csi_survey_count_30d,
                "csi_survey_count_60d": csi_survey_count_60d,
                "csi_survey_count_90d": csi_survey_count_90d,
            }
        )
    return pd.DataFrame(features)


def extract_voc_features(df):
    """Extract VOC and interaction features from RODB data with 30/60/90 day windows"""
    # Define expected columns for empty DataFrame case
    expected_columns = [
        "고객코드",
        "num_interactions",
        "voc_count",
        "non_voc_count",
        "voc_ratio",
        "interactions_30d",
        "voc_30d",
        "non_voc_30d",
        "voc_ratio_30d",
        "interactions_60d",
        "voc_60d",
        "non_voc_60d",
        "voc_ratio_60d",
        "interactions_90d",
        "voc_90d",
        "non_voc_90d",
        "voc_ratio_90d",
    ]

    # Handle empty DataFrame case
    if df.empty:
        return pd.DataFrame(columns=expected_columns)

    features = []
    for idx, row in df.iterrows():
        interactions = (
            json.loads(row["interaction_history"])
            if pd.notna(row["interaction_history"])
            else []
        )
        num_interactions = len(interactions)

        voc_count = sum([1 for i in interactions if i.get("event_type") == "VOC"])
        non_voc_count = num_interactions - voc_count

        interaction_dates = []
        voc_dates = []

        for interaction in interactions:
            date_str = interaction.get("event_date") or interaction.get("date")
            if date_str:
                date = pd.to_datetime(date_str)
                interaction_dates.append(date)
                if interaction.get("event_type") == "VOC":
                    voc_dates.append(date)

        if interaction_dates:
            last_date = max(interaction_dates)

            # Interaction counts by time window
            interactions_30d = sum(
                [1 for d in interaction_dates if (last_date - d).days <= 30]
            )
            interactions_60d = sum(
                [1 for d in interaction_dates if (last_date - d).days <= 60]
            )
            interactions_90d = sum(
                [1 for d in interaction_dates if (last_date - d).days <= 90]
            )

            # VOC counts by time window
            voc_30d = sum([1 for d in voc_dates if (last_date - d).days <= 30])
            voc_60d = sum([1 for d in voc_dates if (last_date - d).days <= 60])
            voc_90d = sum([1 for d in voc_dates if (last_date - d).days <= 90])

            # Non-VOC counts by time window
            non_voc_30d = interactions_30d - voc_30d
            non_voc_60d = interactions_60d - voc_60d
            non_voc_90d = interactions_90d - voc_90d

            # VOC ratios by time window
            voc_ratio_30d = voc_30d / interactions_30d if interactions_30d > 0 else 0
            voc_ratio_60d = voc_60d / interactions_60d if interactions_60d > 0 else 0
            voc_ratio_90d = voc_90d / interactions_90d if interactions_90d > 0 else 0
        else:
            interactions_30d = interactions_60d = interactions_90d = 0
            voc_30d = voc_60d = voc_90d = 0
            non_voc_30d = non_voc_60d = non_voc_90d = 0
            voc_ratio_30d = voc_ratio_60d = voc_ratio_90d = 0

        features.append(
            {
                "고객코드": row["고객코드"],
                # Overall metrics
                "num_interactions": num_interactions,
                "voc_count": voc_count,
                "non_voc_count": non_voc_count,
                "voc_ratio": voc_count / num_interactions
                if num_interactions > 0
                else 0,
                # 30-day window
                "interactions_30d": interactions_30d,
                "voc_30d": voc_30d,
                "non_voc_30d": non_voc_30d,
                "voc_ratio_30d": voc_ratio_30d,
                # 60-day window
                "interactions_60d": interactions_60d,
                "voc_60d": voc_60d,
                "non_voc_60d": non_voc_60d,
                "voc_ratio_60d": voc_ratio_60d,
                # 90-day window
                "interactions_90d": interactions_90d,
                "voc_90d": voc_90d,
                "non_voc_90d": non_voc_90d,
                "voc_ratio_90d": voc_ratio_90d,
            }
        )
    return pd.DataFrame(features)


def add_granular_features(df):
    """Add granular derived features"""
    df = df.copy()

    # Activity ratios - recent activity vs historical
    df["최근90일_활동비율"] = df["work_last_90d"] / (df["work_count"] + 1e-8)
    df["최근30일_활동비율"] = df["work_last_30d"] / (df["work_count"] + 1e-8)

    # Activity density - work per day
    df["활동_밀도"] = df["work_count"] / (df["work_span_days"] + 1)

    # Trend features - comparing recent vs overall rates
    df["확정비율_변화"] = df["confirmation_rate_30d"] - df["confirmation_rate"]
    df["취소비율_변화"] = df["cancellation_rate_30d"] - df["cancellation_rate"]

    # VOC vs Work ratio - customer service interaction intensity
    df["VOC_대_작업비율"] = df["voc_count"] / (df["work_count"] + 1e-8)

    # Recent activity trend - 30d vs 90d comparison
    df["최근30일_대_90일_비율"] = df["work_last_30d"] / (df["work_last_90d"] + 1e-8)

    return df
