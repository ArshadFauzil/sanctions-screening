-- Sanctions Screening Adjudicator — Phase 1 schema (plan section 1.2).
--
-- Loaded into the Postgres instance already running for the aml-adverse-media
-- project (database "ams"). Table names are prefixed sdn_ so they cannot
-- collide with that project's future tables in the same database.
--
-- entry_type is the discriminator (individual / entity / vessel / aircraft)
-- and drives which comparison attributes the Day-2 adjudicator is given.
-- One features table is used for every entry_type rather than a per-type
-- schema: the feature vocabulary differs (dob/nationality for individuals;
-- jurisdiction/registration_number for entities; vessel_flag/vessel_owner
-- for vessels) but the (uid, feature_type, value) shape does not.

CREATE TABLE IF NOT EXISTS sdn_entries (
    uid          INTEGER PRIMARY KEY,           -- OFAC's own uid, stable across publications
    entry_type   TEXT NOT NULL
                 CHECK (entry_type IN ('individual', 'entity', 'vessel', 'aircraft')),
    primary_name TEXT NOT NULL,                 -- "LASTNAME, FIRSTNAME" for individuals, full name otherwise
    first_name   TEXT,                          -- raw XML component, kept for name-order perturbation work
    last_name    TEXT,                          -- raw XML component (holds the full name for non-individuals)
    title        TEXT,
    programs     TEXT[] NOT NULL DEFAULT '{}',
    remarks      TEXT
);

CREATE TABLE IF NOT EXISTS sdn_aliases (
    id         BIGSERIAL PRIMARY KEY,
    uid        INTEGER NOT NULL REFERENCES sdn_entries(uid) ON DELETE CASCADE,
    alias_name TEXT NOT NULL,
    alias_type TEXT,                            -- a.k.a. / f.k.a. / n.k.a.
    is_weak    BOOLEAN NOT NULL DEFAULT FALSE    -- OFAC's own "weak" aka category
);

CREATE TABLE IF NOT EXISTS sdn_features (
    id            BIGSERIAL PRIMARY KEY,
    uid           INTEGER NOT NULL REFERENCES sdn_entries(uid) ON DELETE CASCADE,
    feature_type  TEXT NOT NULL,                -- 'dob' | 'place_of_birth' | 'nationality' |
                                                 -- 'citizenship' | normalized idType (e.g.
                                                 -- 'registration_number', 'swift_bic', 'passport') |
                                                 -- 'vessel_flag' | 'vessel_owner' | ...
    value_text    TEXT,
    value_date    DATE,                         -- populated when value_text parses as a full date
                                                 -- (many OFAC DOBs are year-only or "circa" and stay NULL)
    value_country TEXT                          -- idCountry for id-document features (e.g. passport
                                                 -- issuing country); NULL for everything else
);

CREATE TABLE IF NOT EXISTS sdn_addresses (
    id          BIGSERIAL PRIMARY KEY,
    uid         INTEGER NOT NULL REFERENCES sdn_entries(uid) ON DELETE CASCADE,
    address     TEXT,                           -- address1/2/3 joined
    city        TEXT,
    state       TEXT,                           -- stateOrProvince
    postal_code TEXT,
    country     TEXT
);

CREATE INDEX IF NOT EXISTS idx_sdn_entries_entry_type   ON sdn_entries (entry_type);
CREATE INDEX IF NOT EXISTS idx_sdn_aliases_uid          ON sdn_aliases (uid);
CREATE INDEX IF NOT EXISTS idx_sdn_features_uid         ON sdn_features (uid);
CREATE INDEX IF NOT EXISTS idx_sdn_features_type        ON sdn_features (feature_type);
CREATE INDEX IF NOT EXISTS idx_sdn_addresses_uid        ON sdn_addresses (uid);
CREATE INDEX IF NOT EXISTS idx_sdn_addresses_country    ON sdn_addresses (country);
