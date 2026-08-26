-- ============================================================
-- TTC Transit Service Reliability Analysis — Database Schema
-- ============================================================
-- Star schema: fact_delay_events at the center, with dim_station,
-- dim_date, dim_delay_code, and fact_weather_daily as supporting
-- tables. Tables are created programmatically by ttc_pipeline.py;
-- this file documents the resulting structure.

CREATE TABLE dim_station (
    station_id      INTEGER PRIMARY KEY,
    station_name    TEXT NOT NULL,   -- cleaned canonical name, see normalize_station_name()
    line            TEXT
);

CREATE TABLE dim_delay_code (
    code                TEXT PRIMARY KEY,
    code_description    TEXT,
    category            TEXT
);

CREATE TABLE dim_date (
    date_id         DATE PRIMARY KEY,
    year            INTEGER,
    month           INTEGER,
    day             INTEGER,
    day_of_week     TEXT,
    is_weekend      BOOLEAN,
    is_holiday      BOOLEAN,
    season          TEXT             -- Winter / Spring / Summer / Fall
);

CREATE TABLE fact_weather_daily (
    date_id             DATE PRIMARY KEY REFERENCES dim_date(date_id),
    mean_temp_c         REAL,
    total_precip_mm     REAL,
    total_snow_cm       REAL         -- often NULL depending on weather station
);

CREATE TABLE fact_delay_events (
    date_id         DATE REFERENCES dim_date(date_id),
    time_of_day     TEXT,
    hour            INTEGER,
    station_id      INTEGER REFERENCES dim_station(station_id),
    code            TEXT REFERENCES dim_delay_code(code),
    mode            TEXT,            -- Subway / Bus / Streetcar
    bound           TEXT,
    line            TEXT,
    vehicle_id      TEXT,
    min_delay       REAL,
    min_gap         REAL
);
