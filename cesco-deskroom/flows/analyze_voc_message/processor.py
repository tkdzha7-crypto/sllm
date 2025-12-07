import json
import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class VoCAnalyzer:
    def __init__(self, db_connection, api_url: str = "http://172.16.3.220:8000"):
        self.voc_categories_df = None
        self.categories = None
        self.api_url = api_url
        self.db_connection = db_connection

        # Predefined bug list for TF-IDF matching (kept from ingest.py)
        self.BUG_LIST = [
            "쥐",
            "바퀴",
            "개미",
            "저곡해충",
            "진드기",
            "먼지다듬이",
            "고양이",
            "거미",
            "집게벌레",
            "파리",
            "모기",
            "하루살이",
            "화랑곡나방",
            "나방",
            "미동정",
            "기타",
            "집웅쥐",
            "시궁쥐",
            "생쥐",
            "땃쥐",
            "두더지",
            "들쥐",
            "흰넓적다리붉은쥐",
            "등줄쥐",
            "독일바퀴",
            "미국바퀴",
            "먹바퀴",
            "일본바퀴",
            "경도바퀴",
            "산바퀴",
            "애집개미",
            "침개미",
            "유령개미",
            "미친개미",
            "미동정개미",
            "외곽개미",
            "쌀바구미",
            "팥바구미",
            "거짓쌀도둑거저리",
            "톱가슴머리대장",
            "장두",
            "애알락수시렁이",
            "애수시렁이",
            "암검은수시렁이",
            "권연벌레",
            "거저리",
            "흡혈진드기",
            "쥐며느리",
            "노래기",
            "지네",
            "얼룩점초파리",
            "집파리",
            "딸집파리",
            "나방파리",
            "초파리",
            "얼룩무늬등초파리",
            "날파리",
            "검정날개버섯파리과",
            "벼룩파리",
            "큰검정파리",
            "구리금파리",
            "쉬파리",
            "애기똥파리",
            "붉은등우단털파리",
            "숲모기",
            "지하집모기",
            "장구벌레",
            "깔따구",
            "깔따구유충",
            "줄알락명나방",
            "지중해가루명나방",
            "멸강나방",
            "해충없음",
            "딱정벌레",
            "풍뎅이",
            "먼지벌레",
            "메뚜기",
            "방아깨비",
            "여치",
            "매미",
            "털파리",
            "하늘소",
            "등에",
            "꼽등이",
            "벌",
            "흰개미",
            "멸구",
            "각다귀",
            "사면발이",
            "톡토기",
            "물자라",
            "반날개",
            "풀잠자리",
            "사슴벌레",
            "무당벌레",
            "흡혈해충",
            "벼룩",
            "응애",
            "빈대",
            "보리나방",
            "그리마",
            "귀뚜라미",
            "직물해충",
            "옷좀나방",
            "좀벌레",
            "수목해충",
            "진딧물",
            "송충이",
            "노린재",
            "목재해충",
            "길앞잡이",
            "미국선녀벌레",
            "날도래류",
            "동굴표본벌레",
            "방아벌레",
            "나무좀벌레",
            "권련침벌",
            "가루이",
            "매미충",
            "다듬이벌레(유시충)",
            "썬더블루 포획",
            "블루스톰 포획",
        ]

        # Category name mappings (from ingest.py)
        self.category_mappings = {
            "비용/계약 문제": "요금/계약 문제",
            "계약/비용 문제": "요금/계약 문제",
            "요금 문제": "요금/계약 문제",
            "계약 문제": "요금/계약 문제",
            "해충": "해충 문제",
            "제품 문제": "제품",
            "서비스": "서비스 품질",
            "서비스품질": "서비스 품질",
            "배송": "배송문제",
            "시스템 오류": "시스템/전산 오류",
            "전산 오류": "시스템/전산 오류",
            "운영": "운영 관리",
        }

    def input_text_cleansing(self, text: str) -> str:
        """Cleanses the input text by removing special characters and extra spaces."""
        if not isinstance(text, str):
            return ""
        # Remove special characters (keep Korean, English, numbers, and basic punctuation)
        cleaned_text = re.sub(r"[^가-힣a-zA-Z0-9\s.,!?]", " ", text)
        # Replace multiple spaces with a single space
        cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
        return cleaned_text

    def normalize_category_name(self, category_name: str) -> str:
        """Normalize category name using mapping table"""
        if not category_name:
            return category_name
        if category_name in self.category_mappings:
            normalized = self.category_mappings[category_name]
            # keep a small debug print here for traceability
            print(f"🔄 Mapped '{category_name}' → '{normalized}'")
            return normalized
        return category_name

    def build_input_categories(self):
        """Fetch voc categories from DB and build the nested input_categories structure.
        Structure: {대분류: {중분류: [소분류_list]}}
        The resulting JSON string is stored in self.categories (ready to send to SLLM API).
        """
        # Resolve engine from provided db_connection
        engine = None
        if (
            hasattr(self.db_connection, "engine")
            and self.db_connection.engine is not None
        ):
            engine = self.db_connection.engine
        else:
            engine = self.db_connection

        if engine is None:
            raise RuntimeError("No valid DB engine available in db_connection")

        query = "SELECT id, voc_id, name, parent_id, level FROM source.voc_category"
        try:
            self.voc_categories_df = pd.read_sql(query, engine)
        except Exception as e:
            print(f"❌ Error loading voc_category table: {e}")
            self.voc_categories_df = pd.DataFrame(
                columns=["id", "voc_id", "name", "parent_id", "level"]
            )

        input_categories = {}
        level1_categories = self.voc_categories_df[
            self.voc_categories_df["level"] == 1.0
        ]

        for _, level1_cat in level1_categories.iterrows():
            main_name = level1_cat["name"]
            main_voc_id = level1_cat["voc_id"]

            level2_categories = self.voc_categories_df[
                (self.voc_categories_df["level"] == 2.0)
                & (self.voc_categories_df["parent_id"] == main_voc_id)
            ]

            sub_dict = {}
            for _, level2_cat in level2_categories.iterrows():
                sub_name = level2_cat["name"]
                sub_voc_id = level2_cat["voc_id"]

                level3_categories = self.voc_categories_df[
                    (self.voc_categories_df["level"] == 3.0)
                    & (self.voc_categories_df["parent_id"] == sub_voc_id)
                ]

                detail_list = level3_categories["name"].tolist()
                sub_dict[sub_name] = detail_list

            input_categories[main_name] = sub_dict

        # Store JSON string to send to SLLM
        try:
            self.categories = json.dumps(input_categories, ensure_ascii=False, indent=2)
        except Exception:
            # Fallback to Python dict if JSON dumping fails
            self.categories = input_categories

        print(
            f"✅ Built input_categories structure with {len(input_categories)} main categories"
        )
        return input_categories

    def find_related_bug(self, content_text, threshold=0.3):
        """Find the most related bug using TF-IDF similarity (fallback when SLLM doesn't provide bug_type)."""
        if not content_text or not self.BUG_LIST:
            return None

        try:
            corpus = [content_text] + self.BUG_LIST
            vectorizer = TfidfVectorizer(
                stop_words=None, max_features=1000, ngram_range=(1, 2)
            )
            tfidf_matrix = vectorizer.fit_transform(corpus)
            content_vector = tfidf_matrix[0:1]
            bug_vectors = tfidf_matrix[1:]
            similarities = cosine_similarity(content_vector, bug_vectors).flatten()
            max_similarity = np.max(similarities)
            if max_similarity >= threshold:
                best_bug_index = int(np.argmax(similarities))
                best_bug = self.BUG_LIST[best_bug_index]
                print(
                    f"🔍 TF-IDF found related bug: '{best_bug}' (similarity: {max_similarity:.3f})"
                )
                return best_bug
            else:
                print(
                    f"🔍 No related bug found above threshold {threshold} (max similarity: {max_similarity:.3f})"
                )
                return None
        except Exception as e:
            print(f"⚠️ Error in TF-IDF bug matching: {e}")
            return None

    def get_category_info(self, category_name, level=None, parent_voc_id=None):
        """Get category id, voc_id, and actual name by name, level, and parent relationship. Falls back to '기타' when missing."""
        if not category_name:
            return None, None, None

        normalized_name = self.normalize_category_name(category_name)

        if self.voc_categories_df is None:
            print("⚠️ voc_categories_df not loaded; call build_input_categories() first")
            return None, None, None

        matches = self.voc_categories_df[
            self.voc_categories_df["name"] == normalized_name
        ]
        if level is not None:
            matches = matches[matches["level"] == level]
        if parent_voc_id is not None:
            matches = matches[matches["parent_id"] == parent_voc_id]

        if len(matches) > 0:
            match = matches.iloc[0]
            return int(match["id"]), match["voc_id"], match["name"]
        else:
            parent_info = f" under parent {parent_voc_id}" if parent_voc_id else ""
            print(
                f"⚠️ No match found for category: {normalized_name} (original: {category_name}) (level: {level}){parent_info}"
            )
            print(f"🔄 Falling back to '기타' category for level {level}")

            기타_matches = self.voc_categories_df[
                self.voc_categories_df["name"] == "기타"
            ]
            if level is not None:
                기타_matches = 기타_matches[기타_matches["level"] == level]
            if parent_voc_id is not None:
                기타_matches = 기타_matches[기타_matches["parent_id"] == parent_voc_id]

            if len(기타_matches) > 0:
                기타_match = 기타_matches.iloc[0]
                print(f"✅ Using 기타 category: {기타_match['voc_id']}")
                return int(기타_match["id"]), 기타_match["voc_id"], 기타_match["name"]
            else:
                general_기타_matches = self.voc_categories_df[
                    (self.voc_categories_df["name"] == "기타")
                    & (self.voc_categories_df["level"] == level)
                ]
                if len(general_기타_matches) > 0:
                    기타_match = general_기타_matches.iloc[0]
                    print(f"✅ Using general 기타 category: {기타_match['voc_id']}")
                    return (
                        int(기타_match["id"]),
                        기타_match["voc_id"],
                        기타_match["name"],
                    )
                else:
                    print(f"⚠️ No 기타 category found for level {level}")
                    return None, None, None

    def _postprocess_response(self, sample, sllm_response, confidence_score):
        """Process the SLLM response and build the sample_output dict.

        Args:
            sample: A pandas namedtuple with rcno, ccod, msg_id, received_at, content
            sllm_response: The parsed response from SLLM API
            confidence_score: The confidence score from SLLM API

        Returns:
            dict: The processed sample output with all category mappings
        """
        now_kst = datetime.now(ZoneInfo("Asia/Seoul"))

        if sllm_response is None:
            print("⚠️ Received None response from SLLM API, using default values")
            sllm_response = {
                "categories": [],
                "keywords": None,
                "bug_type": None,
                "is_claim": "no_claim",
                "summary": None,
            }
            confidence_score = 0.0

        inferred_categories = sllm_response.get("categories", [])

        keywords_data = sllm_response.get("keywords")
        if keywords_data is not None:
            filtered_keywords = [kw for kw in keywords_data if kw in sample.content]
            keywords_json = (
                json.dumps(filtered_keywords, ensure_ascii=False)
                if not isinstance(filtered_keywords, str)
                else filtered_keywords
            )
        else:
            keywords_json = None

        bug_type = sllm_response.get("bug_type")
        if not bug_type:
            print(
                f"🔍 No bug_type from SLLM, using TF-IDF for content: {sample.content[:100]}..."
            )
            bug_type = self.find_related_bug(sample.content)

        # The received_at from source is a naive datetime already in KST
        # Keep it as naive so PostgreSQL stores it as-is without conversion
        msg_received_at = sample.received_at
        if hasattr(msg_received_at, "tzinfo") and msg_received_at.tzinfo is not None:
            # If it has timezone info, strip it to prevent double conversion
            msg_received_at = msg_received_at.replace(tzinfo=None)
        elif isinstance(msg_received_at, pd.Timestamp):
            # Convert pandas Timestamp to Python datetime (naive)
            msg_received_at = msg_received_at.to_pydatetime().replace(tzinfo=None)

        sample_output = {
            "rcno": sample.rcno,
            "ccod": sample.ccod,
            "msg_id": sample.msg_id,
            "msg_received_at": msg_received_at
            - timedelta(hours=9),  # Store as UTC naive
            "created_at": now_kst,
            "updated_at": now_kst,
            "model_name": "cesco_sLLM_Qwen_3",
            "model_ver": "1.0",
            "content": sample.content,
            "is_claim": 1 if sllm_response.get("is_claim") == "claim" else 0,
            "summary": sllm_response.get("summary"),
            "keywords": keywords_json,
            "bug_type": bug_type,
            "model_confidence": confidence_score,
        }

        # Map categories (up to 5 levels like ingest.py)
        # First, resolve all categories to get actual names after fallback
        resolved_categories = []
        for i in range(5):
            category_data = (
                inferred_categories[i] if i < len(inferred_categories) else {}
            )

            main_name = category_data.get("대분류")
            sub_name = category_data.get("중분류")
            detail_name = category_data.get("소분류")
            detail_reason = category_data.get("근거")

            main_id, main_code, actual_main_name = self.get_category_info(
                main_name, level=1
            )
            sub_id, sub_code, actual_sub_name = (
                self.get_category_info(sub_name, level=2, parent_voc_id=main_code)
                if main_code
                else (None, None, None)
            )
            detail_id, detail_code, actual_detail_name = (
                self.get_category_info(detail_name, level=3, parent_voc_id=sub_code)
                if sub_code
                else (None, None, None)
            )

            resolved_categories.append(
                {
                    "main_id": main_id,
                    "main_code": main_code,
                    "main_name": actual_main_name or main_name,
                    "sub_id": sub_id,
                    "sub_code": sub_code,
                    "sub_name": actual_sub_name or sub_name,
                    "detail_id": detail_id,
                    "detail_code": detail_code,
                    "detail_name": actual_detail_name or detail_name,
                    "detail_reason": detail_reason,
                }
            )

        # Sort resolved categories: push "기타-기타-기타" to the end
        def is_all_기타_resolved(resolved):
            """Check if all resolved category names are '기타'"""
            if not resolved or (
                resolved["main_code"] is None
                and resolved["sub_code"] is None
                and resolved["detail_code"] is None
            ):
                return True  # Empty categories go to end
            return (
                resolved.get("main_name") == "기타"
                and resolved.get("sub_name") == "기타"
                and resolved.get("detail_name") == "기타"
            )

        def is_empty_resolved(resolved):
            """Check if category is empty (all codes are None)"""
            if not resolved:
                return True
            return (
                resolved["main_code"] is None
                and resolved["sub_code"] is None
                and resolved["detail_code"] is None
            )

        # Separate into: proper categories, 기타-기타-기타, and empty categories
        proper_categories = [
            c
            for c in resolved_categories
            if not is_all_기타_resolved(c) and not is_empty_resolved(c)
        ]
        기타_categories = [
            c
            for c in resolved_categories
            if is_all_기타_resolved(c) and not is_empty_resolved(c)
        ]
        empty_categories = [c for c in resolved_categories if is_empty_resolved(c)]

        # Reorder: proper categories first, then 기타-기타-기타, then empty
        resolved_categories = proper_categories + 기타_categories + empty_categories

        # Track seen category combinations for deduplication
        seen_category_combinations = set()

        for i, resolved in enumerate(resolved_categories):
            level_num = i + 1

            # Create a unique key for this category combination
            category_key = (
                resolved["main_code"],
                resolved["sub_code"],
                resolved["detail_code"],
            )

            # Check for duplicate: skip if already seen or if all codes are None
            is_duplicate = category_key in seen_category_combinations
            is_empty = category_key == (None, None, None)

            if is_duplicate and not is_empty:
                print(
                    f"🔄 Skipping duplicate category combination at level {level_num}: {category_key}"
                )
                sample_output.update(
                    {
                        f"main_category_{level_num}_name": None,
                        f"main_category_{level_num}_id": None,
                        f"main_category_{level_num}_code": None,
                        f"sub_category_{level_num}_name": None,
                        f"sub_category_{level_num}_id": None,
                        f"sub_category_{level_num}_code": None,
                        f"detail_category_{level_num}_name": None,
                        f"detail_category_{level_num}_id": None,
                        f"detail_category_{level_num}_code": None,
                        f"detail_category_{level_num}_reason": None,
                    }
                )
            else:
                # Add to seen set if not empty
                if not is_empty:
                    seen_category_combinations.add(category_key)

                sample_output.update(
                    {
                        f"main_category_{level_num}_name": resolved["main_name"],
                        f"main_category_{level_num}_id": resolved["main_id"],
                        f"main_category_{level_num}_code": resolved["main_code"],
                        f"sub_category_{level_num}_name": resolved["sub_name"],
                        f"sub_category_{level_num}_id": resolved["sub_id"],
                        f"sub_category_{level_num}_code": resolved["sub_code"],
                        f"detail_category_{level_num}_name": resolved["detail_name"],
                        f"detail_category_{level_num}_id": resolved["detail_id"],
                        f"detail_category_{level_num}_code": resolved["detail_code"],
                        f"detail_category_{level_num}_reason": resolved[
                            "detail_reason"
                        ],
                    }
                )

        print(sample_output)
        return sample_output

    def analyze_message_in_batch(self, samples: pd.DataFrame):
        if self.categories is None or self.voc_categories_df is None:
            try:
                self.build_input_categories()
            except Exception as e:
                print(f"⚠️ Failed to build input categories: {e}")

        payload = {
            "input_texts": [
                self.input_text_cleansing(sample.content)
                for _, sample in samples.iterrows()
            ],
            "input_categories": self.categories,
            "max_new_tokens": 512,
            "temperature": 0.1,
            "top_p": 0.9,
        }
        response = requests.post(f"{self.api_url}/batch", json=payload)
        print(f"Batch API Response Status: {response.status_code}")
        if response.status_code == 200:
            parsed_responses = [item.get("parsed_response") for item in response.json()]
            confidence_scores = [
                item.get("confidence_score", 0.0) for item in response.json()
            ]
        else:
            print(f"API Error: {response.text}")
            parsed_responses = [None] * len(samples)
            confidence_scores = [0.0] * len(samples)

        sample_outputs = []
        for i, (_, sample) in enumerate(samples.iterrows()):
            sllm_response = parsed_responses[i]
            confidence_score = confidence_scores[i]

            # Use shared postprocess logic
            sample_output = self._postprocess_response(
                sample, sllm_response, confidence_score
            )
            sample_outputs.append(sample_output)

        return sample_outputs

    def analyze_message(self, sample):
        """Analyze a single sample (pandas namedtuple or similar). Returns the sample_output dict.

        Expects sample to have attributes: rcno, ccod, msg_id, received_at, content
        """
        # Ensure categories and voc categories are built
        if self.categories is None or self.voc_categories_df is None:
            try:
                self.build_input_categories()
            except Exception as e:
                print(f"⚠️ Failed to build input categories: {e}")

        payload = {
            "input_text": self.input_text_cleansing(sample.content),
            "input_categories": self.categories,
            "max_new_tokens": 512,
            "temperature": 0.1,
            "top_p": 0.9,
        }

        response = requests.post(f"{self.api_url}/predict", json=payload)
        print(f"Sample RCNO: {sample.rcno}")
        print(f"API Response Status: {response.status_code}")
        if response.status_code == 200:
            response_json = response.json()
            sllm_response = response_json.get("parsed_response")
            confidence_score = response_json.get("confidence_score", 0.0)
            print(f"SLLM Response: {sllm_response}")
        else:
            print(f"API Error: {response.text}")
            sllm_response = None
            confidence_score = 0.0

        # Use shared postprocess logic
        return self._postprocess_response(sample, sllm_response, confidence_score)
