ride_files = ['data/raw/7_dwarfs_train.csv', 'data/raw/astro_orbiter.csv', 'data/raw/barnstormer.csv',
              'data/raw/big_thunder_mtn.csv', 'data/raw/buzz_lightyear.csv',
              'data/raw/carousel_of_progress.csv', 'data/raw/dumbo.csv',
              'data/raw/haunted_mansion.csv', 'data/raw/it_s_a_small_world.csv',
              'data/raw/jungle_cruise.csv', 'data/raw/mad_tea_party.csv', 'data/raw/magic_carpets.csv',
              'data/raw/main_st_vehicles.csv', 'data/raw/peoplemover.csv',
              'data/raw/peter_pan_s_flight.csv',
              'data/raw/pirates_of_caribbean.csv', 'data/raw/regal_carrousel.csv',
              'data/raw/space_mountain.csv',
              'data/raw/splash_mountain.csv', 'data/raw/tom_land_speedway.csv',
              'data/raw/winnie_the_pooh.csv']

ride_names = ['Seven Dwarfs Mine Train', 'Astro Orbiter', 'The Barnstormer', 'Big Thunder Mountain Railroad',
              "Buzz Lightyear's Space Ranger Spin", "Walt Disney's Carousel of Progress",
              'Dumbo the Flying Elephant',
              'Haunted Mansion', "It's a Small World", 'Jungle Cruise', 'Mad Tea Party',
              'The Magic Carpets of Aladdin',
              'Main Street Vehicles', 'Tomorrowland Transit Authority PeopleMover', "Peter Pan's Flight",
              'Pirates of the Caribbean', 'Prince Charming Regal Carrousel', 'Space Mountain', 'Splash Mountain',
              'Tomorrowland Speedway', 'The Many Adventures of Winnie the Pooh']

categoricalCols = ["WDW_TICKET_SEASON", "SEASON", "HOLIDAYN",
                           "WDWRaceN", "WDWeventN", "WDWSEASON",
                           "MKeventN", "EPeventN", "HSeventN", "AKeventN",
                           "HOLIDAYJ", "Ride_name", "Park_area",
                           "MKPRDDN", "MKPRDNN", "MKFIREN",
                           "EPFIREN", "HSPRDDN", "HSFIREN",
                           "HSSHWNN", "AKPRDDN", "AKFIREN", "AKSHWNN",
                           "Wind Quality Code", "Wind Type Code", "Wind Speed Quality",
                            "Cloud Quality Code", "Cloud Determination Code", "CAVOK Code",
                            "Visibiliy Quality Code", "Visibility Variability Code",
                            "Visibility Quality Variability Code", "Temperature Quality Code"]
bool_dtypes = [
    "Ride_type_thrill", "Ride_type_spinning", "Ride_type_slow",
    "Ride_type_small_drops", "Ride_type_big_drops", "Ride_type_dark",
    "Ride_type_scary", "Ride_type_water", "Fast_pass",
    "Classic", "Age_interest_preschoolers", "Age_interest_kids",
    "Age_interest_tweens", "Age_interest_teens", "Age_interest_adults",
    "HOLIDAY", "WDWevent", "WDWrace", "MKevent", "EPevent", "HSevent",
    "AKevent", "MKEMHMORN", "MKEMHMYEST", "MKEMHMTOM", "MKEMHEVE",
    "MKEMHEYEST","MKEMHETOM", "EPEMHMORN", "EPEMHMYEST","EPEMHMTOM",
    "EPEMHEVE", "EPEMHEYEST", "EPEMHETOM",   "HSEMHMORN", "HSEMHMYEST",
    "HSEMHMTOM", "HSEMHEVE", "HSEMHEYEST", "HSEMHETOM",  "AKEMHMORN",
    "AKEMHMYEST", "AKEMHMTOM", "AKEMHEVE", "AKEMHEYEST", "AKEMHETOM"
]

parse_dates = ['date', 'datetime']
parse_times = ["MKOPEN", "MKCLOSE", "MKEMHOPEN", "MKEMHCLOSE",
               "MKOPENYEST", "MKCLOSEYEST", "MKOPENTOM",
               "MKCLOSETOM","EPOPEN", "EPCLOSE", "EPEMHOPEN",
               "EPEMHCLOSE", "EPOPENYEST", "EPCLOSEYEST",
               "EPOPENTOM", "EPCLOSETOM", "HSOPEN", "HSCLOSE",
               "HSEMHOPEN", "HSEMHCLOSE", "HSOPENYEST", "HSCLOSEYEST",
               "HSOPENTOM", "HSCLOSETOM", "AKOPEN", "AKCLOSE",
               "AKEMHOPEN", "AKOPENYEST", "AKCLOSEYEST",
               "AKOPENTOM", "AKCLOSETOM", "MKPRDDT1", "MKPRDDT2",
               "MKPRDNT1", "MKPRDNT2", "MKFIRET1", "MKFIRET2",
               "EPFIRET1", "EPFIRET2", "HSPRDDT1", "HSFIRET1",
               "HSFIRET2", "HSSHWNT1", "HSSHWNT2", "AKPRDDT1",
               "AKPRDDT2", "AKSHWNT1", "AKSHWNT2"]


park_metadata_cols = ["WEEKOFYEAR","SEASON", "HOLIDAYPX", "HOLIDAYM", "HOLIDAYN", "HOLIDAY",
                    "WDWRaceN", "WDWeventN", "WDWevent",
                    "WDWrace", "WDWSEASON", "WDWMAXTEMP",
                    "WDWMINTEMP", "WDWMEANTEMP", "MKeventN",
                    "MKevent", "EPeventN", "EPevent",
                    "HSeventN", "HSevent", "AKeventN", "AKevent",
                    "HOLIDAYJ", "inSession", "inSession_Enrollment", "inSession_wdw"]
