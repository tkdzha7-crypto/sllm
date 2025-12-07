-- Create databases first
SELECT 'CREATE DATABASE deskroom_core' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'deskroom_core')\gexec
SELECT 'CREATE DATABASE prefect_db' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'prefect_db')\gexec

-- Connect to deskroom_core database for the schema creation
\c deskroom_core;

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE SCHEMA IF NOT EXISTS source;

CREATE TABLE IF NOT EXISTS source.message (
    id BIGSERIAL PRIMARY KEY,
    RCNO VARCHAR(50),
    received_at TIMESTAMPTZ DEFAULT NOW(),
    CCOD VARCHAR(50),
    content TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);


create table if not exists source.user_monthly_features (
	snapshot_id BIGSERIAL,
	CCOD VARCHAR(50),
	snapshot_month DATE not null, -- The partition key
	contract_info JSONB,
	user_information JSONB,
	purchase_logs JSONB,
	interaction_history JSONB,
	PRIMARY KEY (snapshot_id, snapshot_month)
) partition by range (snapshot_month);


create table if not exists source.industry_codes (
	id BIGSERIAL PRIMARY KEY,
	code VARCHAR(50),
	name VARCHAR(255),
	created_at TIMESTAMPTZ DEFAULT NOW(),
	updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE SCHEMA IF NOT EXISTS analytics;
CREATE SCHEMA IF NOT EXISTS monitoring;

-- VOC Category lookup table
CREATE TABLE IF NOT EXISTS source.voc_category (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    voc_id VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    parent_id VARCHAR(50),
    level INTEGER,
    FOREIGN KEY (parent_id) REFERENCES source.voc_category(voc_id)
);

CREATE TABLE IF NOT EXISTS monitoring.model_health (
	id BIGSERIAL PRIMARY KEY,
	model_name VARCHAR(100),
	model_version VARCHAR(50),
	checked_at TIMESTAMPTZ DEFAULT NOW(),
	unknown_rate FLOAT,
	correction_rate FLOAT,
	confidence_drift FLOAT
);

CREATE TABLE IF NOT EXISTS source.potential_user (
	id BIGSERIAL,
	snapshot_month DATE NOT NULL, -- The partition key
	KEDCD VARCHAR(50),
	BZPL_CD VARCHAR(50),
	BZPL_SEQ VARCHAR(50),
	BZNO VARCHAR(50) NOT NULL,
	ENP_NM VARCHAR(255),
	created_at TIMESTAMPTZ DEFAULT NOW(),
	SIDO VARCHAR(100),
	SIGUNGU VARCHAR(100),
	LAT FLOAT,
	LOT FLOAT,
	BZPL_NM VARCHAR(255),
	BF_BZC_CD VARCHAR(50),
	BSZE_METR FLOAT,
	LSZE_METR FLOAT,
	RDNM_ADDR VARCHAR(500),
	PRIMARY KEY (id, snapshot_month)
) PARTITION BY RANGE (snapshot_month);

CREATE TABLE IF NOT EXISTS analytics.potential_user_recommendation (
	id BIGSERIAL,
	enp_nm VARCHAR(255),
	created_at TIMESTAMPTZ DEFAULT NOW(),
	snapshot_month DATE NOT NULL, -- The partition key
	KEDCD VARCHAR(50),
	BZPL_CD VARCHAR(50),
	BZPL_SEQ VARCHAR(50),
	BZNO VARCHAR(50) NOT NULL,
	ENP_NP VARCHAR(255),
	rec_contract_1 VARCHAR(255),
	rec_contract_1_reason TEXT,
	rec_contract_2 VARCHAR(255),
	rec_contract_2_reason TEXT,
	rec_contract_3 VARCHAR(255),
	rec_contract_3_reason TEXT,
	rec_product_1 VARCHAR(255),
	rec_product_1_reason TEXT,
	rec_product_2 VARCHAR(255),
	rec_product_2_reason TEXT,
	rec_product_3 VARCHAR(255),
	rec_product_3_reason TEXT,
	user_cluster INT,
	cluster_similarity FLOAT,
	sim_CCOD VARCHAR(50),
	sim_user_name VARCHAR(255),
	sim_user_contracts JSONB,
	sim_user_products JSONB,
	PRIMARY KEY (id, snapshot_month)
) partition by range (snapshot_month);


create table if not exists analytics.user_churn_prediction (
	id BIGSERIAL,
	created_at TIMESTAMPTZ default NOW(),
	snapshot_month DATE not null, -- The partition key
	CCOD VARCHAR(50),
	churn_prob FLOAT,
	churn_label INT, -- (1: 해약, 0: 잔존)
	feature_1_kor VARCHAR(50),
	feature_1_value FLOAT,
	feature_1_normal_average FLOAT,
	feature_1_churn_average FLOAT,
	feature_1_contrib FLOAT,
	feature_1_contrib_label VARCHAR(50),
	feature_2_kor VARCHAR(50),
	feature_2_value FLOAT,
	feature_2_normal_average FLOAT,
	feature_2_churn_average FLOAT,
	feature_2_contrib FLOAT,
	feature_2_contrib_label VARCHAR(50),
	feature_3_kor VARCHAR(50),
	feature_3_value FLOAT,
	feature_3_normal_average FLOAT,
	feature_3_churn_average FLOAT,
	feature_3_contrib FLOAT,
	feature_3_contrib_label VARCHAR(50),
	feature_4_kor VARCHAR(50),
	feature_4_value FLOAT,
	feature_4_normal_average FLOAT,
	feature_4_churn_average FLOAT,
	feature_4_contrib FLOAT,
	feature_4_contrib_label VARCHAR(50),
	feature_5_kor VARCHAR(50),
	feature_5_value FLOAT,
	feature_5_normal_average FLOAT,
	feature_5_churn_average FLOAT,
	feature_5_contrib FLOAT,
	feature_5_contrib_label VARCHAR(50),
	feature_6_kor VARCHAR(50),
	feature_6_value FLOAT,
	feature_6_normal_average FLOAT,
	feature_6_churn_average FLOAT,
	feature_6_contrib FLOAT,
	feature_6_contrib_label VARCHAR(50),
	feature_7_kor VARCHAR(50),
	feature_7_value FLOAT,
	feature_7_normal_average FLOAT,
	feature_7_churn_average FLOAT,
	feature_7_contrib FLOAT,
	feature_7_contrib_label VARCHAR(50),
	feature_8_kor VARCHAR(50),
	feature_8_value FLOAT,
	feature_8_normal_average FLOAT,
	feature_8_churn_average FLOAT,
	feature_8_contrib FLOAT,
	feature_8_contrib_label VARCHAR(50),
	feature_9_kor VARCHAR(50),
	feature_9_value FLOAT,
	feature_9_normal_average FLOAT,
	feature_9_churn_average FLOAT,
	feature_9_contrib FLOAT,
	feature_9_contrib_label VARCHAR(50),
	feature_10_kor VARCHAR(50),
	feature_10_value FLOAT,
	feature_10_normal_average FLOAT,
	feature_10_churn_average FLOAT,
	feature_10_contrib FLOAT,
	feature_10_contrib_label VARCHAR(50),

	PRIMARY KEY (id, snapshot_month)
) partition by range (snapshot_month);


CREATE TABLE IF NOT EXISTS analytics.user_recommendation (
	id BIGSERIAL,
	created_at TIMESTAMPTZ DEFAULT NOW(),
	CCOD VARCHAR(50) NOT NULL,
	snapshot_month DATE NOT NULL,
	rec_contract_1 VARCHAR(255),
	rec_contract_1_reason TEXT,
	rec_contract_2 VARCHAR(255),
	rec_contract_2_reason TEXT,
	rec_contract_3 VARCHAR(255),
	rec_contract_3_reason TEXT,
	rec_product_1 VARCHAR(255),
	rec_product_1_reason TEXT,
	rec_product_2 VARCHAR(255),
	rec_product_2_reason TEXT,
	rec_product_3 VARCHAR(255),
	rec_product_3_reason TEXT,
	user_cluster INT,
	cluster_similarity FLOAT,
	sim_CCOD VARCHAR(50),
	sim_user_name VARCHAR(255),
	sim_user_contracts JSONB,
	sim_user_products JSONB,
	PRIMARY KEY (id, snapshot_month)
) partition by range (snapshot_month);


CREATE TABLE IF NOT EXISTS analytics.cluster_profile (
	id BIGSERIAL,
	snapshot_month DATE NOT NULL,
	created_at TIMESTAMPTZ DEFAULT NOW(),
	cluster_id INT,
	cluster_size INT,
	avg_contracts_num FLOAT,
	avg_purchase_num FLOAT,
	top_contracts JSONB,
	top_purchases JSONB,
	top_business_type JSONB,
	top_first_contract_code JSONB,
	contracts_distribution JSONB,
	purchase_distribution JSONB,
	PRIMARY KEY (id, snapshot_month)
) partition by range (snapshot_month);


CREATE TABLE IF NOT EXISTS analytics.user_voc_activity (
    id BIGSERIAL PRIMARY KEY,
    aggregate_date DATE NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    ccod VARCHAR(50) NOT NULL,
    category_code VARCHAR(50) NOT NULL,
    category_name VARCHAR(255),
    recontact_agg_day INT DEFAULT 0,
    recontact_past_24h INT DEFAULT 0,
    recontact_past_3d INT DEFAULT 0,
    recontact_past_7d INT DEFAULT 0,
    recontact_past_30d INT DEFAULT 0,

    -- Add unique constraint to prevent duplicates
    UNIQUE(aggregate_date, ccod, category_code)
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_user_voc_activity_date ON analytics.user_voc_activity(aggregate_date);
CREATE INDEX IF NOT EXISTS idx_user_voc_activity_ccod ON analytics.user_voc_activity(ccod);
CREATE INDEX IF NOT EXISTS idx_user_voc_activity_category ON analytics.user_voc_activity(category_code);

CREATE TABLE IF NOT EXISTS analytics.voc_message_category (
	id BIGSERIAL PRIMARY KEY,
	msg_id INT,
	ccod VARCHAR(50),
	RCNO VARCHAR(50),
	msg_received_at TIMESTAMPTZ,
	created_at TIMESTAMPTZ DEFAULT NOW(),
	updated_at TIMESTAMPTZ DEFAULT NOW(),
	model_name VARCHAR(50),
	model_ver VARCHAR(50),
	content TEXT,
	is_claim INT,
	summary TEXT,
	keywords JSONB,
	bug_type VARCHAR(50),

	-- Category Level 1
	main_category_1_name VARCHAR(50),
	main_category_1_id INT,
	main_category_1_code VARCHAR(50),
	sub_category_1_name VARCHAR(50),
	sub_category_1_id INT,
	sub_category_1_code VARCHAR(50),
	detail_category_1_name VARCHAR(50),
	detail_category_1_id INT,
	detail_category_1_code VARCHAR(50),
	detail_category_1_reason TEXT,

	-- Category Level 2
	main_category_2_name VARCHAR(50),
	main_category_2_id INT,
	main_category_2_code VARCHAR(50),
	sub_category_2_name VARCHAR(50),
	sub_category_2_id INT,
	sub_category_2_code VARCHAR(50),
	detail_category_2_name VARCHAR(50),
	detail_category_2_id INT,
	detail_category_2_code VARCHAR(50),
	detail_category_2_reason TEXT,

	-- Category Level 3
	main_category_3_name VARCHAR(50),
	main_category_3_id INT,
	main_category_3_code VARCHAR(50),
	sub_category_3_name VARCHAR(50),
	sub_category_3_id INT,
	sub_category_3_code VARCHAR(50),
	detail_category_3_name VARCHAR(50),
	detail_category_3_id INT,
	detail_category_3_code VARCHAR(50),
	detail_category_3_reason TEXT,

	-- Category Level 4
	main_category_4_name VARCHAR(50),
	main_category_4_id INT,
	main_category_4_code VARCHAR(50),
	sub_category_4_name VARCHAR(50),
	sub_category_4_id INT,
	sub_category_4_code VARCHAR(50),
	detail_category_4_name VARCHAR(50),
	detail_category_4_id INT,
	detail_category_4_code VARCHAR(50),
	detail_category_4_reason TEXT,

	-- Category Level 5
	main_category_5_name VARCHAR(50),
	main_category_5_id INT,
	main_category_5_code VARCHAR(50),
	sub_category_5_name VARCHAR(50),
	sub_category_5_id INT,
	sub_category_5_code VARCHAR(50),
	detail_category_5_name VARCHAR(50),
	detail_category_5_id INT,
	detail_category_5_code VARCHAR(50),
	detail_category_5_reason TEXT,

	-- Confidence scores
	model_confidence FLOAT,

	-- Foreign key constraints
	FOREIGN KEY (msg_id) REFERENCES source.message(id),
	FOREIGN KEY (main_category_1_id) REFERENCES source.voc_category(id),
	FOREIGN KEY (main_category_2_id) REFERENCES source.voc_category(id),
	FOREIGN KEY (main_category_3_id) REFERENCES source.voc_category(id),
	FOREIGN KEY (main_category_4_id) REFERENCES source.voc_category(id),
	FOREIGN KEY (main_category_5_id) REFERENCES source.voc_category(id),
	FOREIGN KEY (sub_category_1_id) REFERENCES source.voc_category(id),
	FOREIGN KEY (sub_category_2_id) REFERENCES source.voc_category(id),
	FOREIGN KEY (sub_category_3_id) REFERENCES source.voc_category(id),
	FOREIGN KEY (sub_category_4_id) REFERENCES source.voc_category(id),
	FOREIGN KEY (sub_category_5_id) REFERENCES source.voc_category(id),
	FOREIGN KEY (detail_category_1_id) REFERENCES source.voc_category(id),
	FOREIGN KEY (detail_category_2_id) REFERENCES source.voc_category(id),
	FOREIGN KEY (detail_category_3_id) REFERENCES source.voc_category(id),
	FOREIGN KEY (detail_category_4_id) REFERENCES source.voc_category(id),
	FOREIGN KEY (detail_category_5_id) REFERENCES source.voc_category(id)
);

CREATE TABLE analytics.user_monthly_property (
    ccod VARCHAR(255) NOT NULL,
    created_at TIMESTAMP NOT NULL,
    unique_work_types INTEGER,
    cancelled_work INTEGER,
    cancellation_rate REAL,
    confirmation_rate REAL,
    avg_services_per_work REAL,
    work_last_30d INTEGER,
    confirmed_30d INTEGER,
    cancelled_30d INTEGER,
    services_30d INTEGER,
    work_types_30d INTEGER,
    cancellation_rate_30d REAL,
    confirmation_rate_30d REAL,
    work_last_60d INTEGER,
    confirmed_60d INTEGER,
    cancelled_60d INTEGER,
    services_60d INTEGER,
    work_types_60d INTEGER,
    cancellation_rate_60d REAL,
    confirmation_rate_60d REAL,
    work_last_90d INTEGER,
    confirmed_90d INTEGER,
    cancelled_90d INTEGER,
    services_90d INTEGER,
    work_types_90d INTEGER,
    cancellation_rate_90d REAL,
    confirmation_rate_90d REAL,
    avg_csi_score REAL,
    csi_score_30d REAL,
    csi_score_60d REAL,
    csi_score_90d REAL,
    csi_survey_count INTEGER,
    csi_survey_count_30d INTEGER,
    csi_survey_count_60d INTEGER,
    csi_survey_count_90d INTEGER,
    num_interactions INTEGER,
    voc_count INTEGER,
    non_voc_count INTEGER,
    voc_ratio REAL,
    interactions_30d INTEGER,
    voc_30d INTEGER,
    non_voc_30d INTEGER,
    voc_ratio_30d REAL,
    interactions_60d INTEGER,
    voc_60d INTEGER,
    non_voc_60d INTEGER,
    voc_ratio_60d REAL,
    interactions_90d INTEGER,
    voc_90d INTEGER,
    non_voc_90d INTEGER,
    voc_ratio_90d REAL,
    recent_90d_activity_ratio REAL,
    recent_30d_activity_ratio REAL,
    activity_density REAL,
    confirmation_rate_change REAL,
    cancellation_rate_change REAL,
    voc_to_work_ratio REAL,
    recent_30d_to_90d_ratio REAL,
    PRIMARY KEY (ccod, created_at)
) PARTITION BY RANGE (created_at);
