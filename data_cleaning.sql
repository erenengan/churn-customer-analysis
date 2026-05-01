BEGIN TRANSACTION
/*
brand_health
Step 1: Convert Trial, P1M, P3M, Awareness, Spontaneous, Brand_Likability, Weekly, Daily into binary
Step 2: Replace NULL values in Fre#visit, PPA, NPS#P3M
Step 3: Encode comprehension levels into numeric scalce (0-4)
Step 4: Assign default NPS Group for missing values: NPS#P3M#Group → 'Detractor' when NULL
Step 5: Drop unnecessary column: spending_use
*/

UPDATE brand_health
SET Trial = 
    CASE
        WHEN Trial = Brand AND Trial IS NOT NULL THEN '1'
        ELSE '0'
    END

UPDATE brand_health
SET P1M = 
    CASE
        WHEN P1M = Brand AND P1M IS NOT NULL THEN '1'
        ELSE '0'
    END

UPDATE brand_health
SET P3M = 
    CASE
        WHEN P3M = Brand AND P3M IS NOT NULL THEN '1'
        ELSE '0'
    END

UPDATE brand_health
SET Awareness = 
    CASE
        WHEN Awareness = Brand AND Awareness IS NOT NULL THEN '1'
        ELSE '0'
    END

UPDATE brand_health
SET Spontaneous = 
    CASE
        WHEN Spontaneous = Brand AND Spontaneous IS NOT NULL THEN '1'
        ELSE '0'
    END

UPDATE brand_health
SET Brand_Likability = 
    CASE
        WHEN Brand_Likability = Brand AND Brand_Likability IS NOT NULL THEN '1'
        ELSE '0'
    END

UPDATE brand_health
SET Weekly = 
    CASE
        WHEN Weekly = Brand AND Weekly IS NOT NULL THEN '1'
        ELSE '0'
    END

UPDATE brand_health
SET Daily = 
    CASE
        WHEN Daily = Brand AND Daily IS NOT NULL THEN '1'
        ELSE '0'
    END

UPDATE brand_health
SET Fre#visit = 0
WHERE Fre#visit IS NULL

UPDATE brand_health
SET Comprehension = 
    CASE
        WHEN Comprehension = 'Do not know it at all' THEN '0'
        WHEN Comprehension = 'Maybe do not know it' THEN '1'
        WHEN Comprehension = 'Know a little' THEN '2'
        WHEN Comprehension = 'Know it well' THEN '3'
        WHEN Comprehension = 'Know it very well' THEN '4'
        ELSE '0'
    END


UPDATE brand_health
SET PPA = 0
WHERE PPA IS NULL

UPDATE brand_health
SET NPS#P3M = 0
WHERE NPS#P3M IS NULL

UPDATE brand_health
SET NPS#P3M#Group = 'Detractor'
WHERE NPS#P3M = '0'

ALTER TABLE brand_health
DROP COLUMN spending_use

/*
brand_image
Step 1: Classify attributes into strategic categories
Step 2: Drop unnecessary column: BrandImage
Step 3: Delete rows where Awareness is NULL (the mising percentage is roughly 0.05%)
*/


ALTER TABLE brand_image
ADD Attribute_Category VARCHAR(100)

UPDATE brand_image
SET Attribute_Category =
    CASE 
        -- Product Quality
        WHEN Attribute IN (
            'Delicious food',
            'Good coffee taste',
            'Good ice-blended taste',
            'Good tea taste',
            'Good other beverages (other than coffee)'
        ) THEN 'Product Quality'

        -- Environment & Atmosphere
        WHEN Attribute IN (
            'Clean',
            'Comfortable and relaxing environment',
            'Good music',
            'Good place for relaxing',
            'Nice environment design'
        ) THEN 'Environment & Atmosphere'

        -- Convenience
        WHEN Attribute IN (
            'Convenient location'
        ) THEN 'Convenience'

        -- Social & Functional Usage
        WHEN Attribute IN (
            'Feel I belong here',
            'Good place for socializing',
            'Good place for socializing with colleagues',
            'Good place for socializing with family',
            'Good place for socializing with friends',
            'Good place for studying',
            'Good place for working / business meeting',
            'Good place for working / studying'
        ) THEN 'Social & Functional Usage'

        -- Service Experience
        WHEN Attribute IN (
            'Friendly staff',
            'Quick speed of service'
        ) THEN 'Service Experience'

        -- Value
        WHEN Attribute IN (
            'Good value for money'
        ) THEN 'Value'

        -- Digital & Infrastructure
        WHEN Attribute IN (
            'High speed of internet'
        ) THEN 'Digital & Infrastructure'

        -- Menu & Innovation
        WHEN Attribute IN (
            'Diversified menu',
            'Have new product regularly'
        ) THEN 'Menu & Innovation'

        -- Brand Perception
        WHEN Attribute IN (
            'Popular brand',
            'Trusted brand',
            'Recommended by others'
        ) THEN 'Brand Perception'

        ELSE 'Other'
    END

ALTER TABLE brand_image
DROP COLUMN BrandImage

DELETE FROM brand_image
WHERE Awareness IS NULL



/*
sa_var
Step 1: Drop redundant columns: MPI_Mean_Use and MPI#2
Step 2: Assess data quality for occupation grouping: total records, null counts, percentage of null values
Step 3: Map detailed occupation values into broader categories
*/


ALTER TABLE sa_var
DROP COLUMN MPI_Mean_Use


ALTER TABLE sa_var
DROP COLUMN MPI#2

UPDATE sa_var
SET Occupation#Group =
    CASE
        -- None Working
        WHEN Occupation IN (
            'Retirement',
            'Housewife',
            'Job hunting',
            'Unemployed',
            'Pupil / Student'
        ) THEN 'None Working'

        -- Self Employed - Small Business and Freelance
        WHEN Occupation IN (
            'Small Business (small shop owner',
            'Freelance',
            'Broker/ Service provider with no employee'
        ) THEN 'Self Employed - Small Business and Freelance'

        -- Self Employed - Company Owner
        WHEN Occupation IN (
            'Business Owner with less than 10 employees',
            'Self Employed - Company owner (10 - 20 employees)',
            'Self Employed  - Company owner (under 10 employees)'
        ) THEN 'Self Employed - Company Owner'

        -- White Collar
        WHEN Occupation IN (
            'Professional (doctor',
            'Lecturer / Teacher',
            'Civil servant - Staff level',
            'Civil servant  - Senior Management',
            'Civil servant  - Middle Management',
            'Officer - Staff level',
            'Officer - Middle Management',
            'Officer - Senior Management',
            'Junior Manager / Executive'
        ) THEN 'White Collar'

        -- Other Occupations
        WHEN Occupation IN (
            'Semi-skilled labor (salesperson',
            'Skilled Labor (tailor',
            'Unskilled Labor (worker',
            'Agriculture / Forestry (Fishing',
            'Artist (actor/actress',
            'Military / Police',
            'Other'
        ) THEN 'Other Occupations'

        -- Refuse
        WHEN Occupation = 'Refuse' THEN 'Refuse'

        ELSE NULL
    END

/*
coffe_visit
Step 1: Create aggregated visit-level table at ID level combining all day-of-week records
Step 2: Compute total_visit as total sum of Visit#Dayofweek per ID
Step 3: Compute weekday_visit and weekend_visit using conditional aggregation
Step 4: Assign representative City per ID (assumes one dominant or unique city per ID)
*/
CREATE TABLE coffee_visit (
    ID INT,
    total_visit INT,
    City VARCHAR(100),
    weekday_visit INT,
    weekend_visit INT,
    PRIMARY KEY (ID)
)

INSERT INTO coffee_visit (ID, total_visit, City, weekday_visit, weekend_visit)
SELECT 
    d.ID,
    SUM(d.[Visit#Dayofweek]) AS total_visit,
    MAX(d.City) AS City,
    SUM(CASE 
            WHEN d.[Weekday#end] = 'Weekdays' THEN d.[Visit#Dayofweek] 
            ELSE 0 
        END) AS weekday_visit,
    SUM(CASE 
            WHEN d.[Weekday#end] = 'Weekends' THEN d.[Visit#Dayofweek] 
            ELSE 0 
        END) AS weekend_visit
FROM day_of_week d
GROUP BY d.ID

/*
day_of_week

Step 1: Replace NULL values in Visit#Dayofweek with 0
Step 2: Standardize missing categorical values in Dayofweek and Weekday#end as 'Not specified'
Step 3: Remove original Dayofweek table after transformation (data already aggregated into coffee_visit)
*/

UPDATE day_of_week
SET Visit#Dayofweek = 0
WHERE Visit#Dayofweek IS NULL

UPDATE day_of_week
SET Dayofweek = 'Not specified'
WHERE Dayofweek IS NULL

UPDATE day_of_week
SET Weekday#end = 'Not specified'
WHERE Weekday#end IS NULL

DROP TABLE Dayofweek

/*
day_part
Step 1: Replace NULL values in Visit#Daypart with 0
Step 2: Standardize missing Daypart values as 'Not specified' for consistency in analysis
*/
UPDATE day_part
SET Visit#Daypart = 0
WHERE Visit#Daypart IS NULL

UPDATE day_part
SET Daypart = 'Not specified'
WHERE Daypart IS NULL

/*
need_state
Step 1: Map detailed Needstates into broader NeedstateGroup categories for analysis
Step 2: Ensure no NULL categories remain by identifying and handling unmatched values
Step 3: Standardize missing Day#Daypart values as 'Not specified' for completeness
*/

UPDATE need_state
SET NeedstateGroup =
    CASE 
        WHEN Needstates IN ('Have meals (breakfast / lunch / dinner)', 'Have snack / pastry')
            THEN 'Meals & Snack'

        WHEN Needstates IN ('Drinking tea', 'Drinking coffee', 
                            'Drinking other beverages (excluding tea', 'Drinking ice-blended')
            THEN 'Drinking beverages'

        WHEN Needstates IN ('Relaxing (Alone)', 
                            'Enterntainment (watching movies. Playing games')
            THEN 'Relaxing & entertainment'

        WHEN Needstates IN ('Socializing with colleagues', 
                            'Socializing with family / relatives',
                            'Socializing with friends', 
                            'Socialzing')
            THEN 'Socializing'

        WHEN Needstates = 'Working / Business meeting'
            THEN 'Working & business meeting'

        WHEN Needstates IN ('Studying / Reading books', 'Other')
            THEN 'Studying & Others'

        ELSE NULL
    END

UPDATE need_state
SET Day#Daypart = 'Not specified'
WHERE Day#Daypart IS NULL


    COMMIT