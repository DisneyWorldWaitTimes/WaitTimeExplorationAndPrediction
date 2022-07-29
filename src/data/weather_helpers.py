# Dictionaries to convert rows in df to more human readable ones
WQCDict = {"0": "Passed gross limits check", "1": "Passed all quality control checks",
           "2": "Suspect",
           "3": "Erroneous",
           "4": "Passed gross limits check, data originate from an NCEI data source",
           "5": "Passed all quality control checks, data originate from an NCEI data source",
           "6": "Suspect, data originate from an NCEI data source",
           "7": "Erroneous, data originate from an NCEI data source",
           "9": "Passed gross limits check if element is present"}

WSQDict = {"0": "Passed gross limits check", "1": "Passed all quality control checks", "2": "Suspect",
           "3": "Erroneous", "4": "Passed gross limits check, data originate from an NCEI data source",
           "5": "Passed all quality control checks, data originate from an NCEI data source",
           "6": "Suspect, data originate from an NCEI data source",
           "7": "Erroneous, data originate from an NCEI data source",
           "9": "Passed gross limits check if element is present"}

WTCDict = {"A": "Abridged Beaufort", "B": "Beaufort", "C": "Calm", "H": "5-Minute Average Speed", "N": "Normal",
           "R": "60-Minute Average Speed", "Q": "Squall", "T": "180 Minute Average Speed", "V": "Variable",
           "9": "Missing"}

CQCDict = {"0": "Passed gross limits check", "1": "Passed all quality control checks", "2": "Suspect",
           "3": "Erroneous", "4": "Passed gross limits check, data originate from an NCEI data source",
           "5": "Passed all quality control checks, data originate from an NCEI data source",
           "6": "Suspect, data originate from an NCEI data source",
           "7": "Erroneous, data originate from an NCEI data source",
           "9": "Passed gross limits check if element is present"}

CDCDict = {"A": "Aircraft", "B": "Balloon", "C": "Statistically derived",
           "D": "Persistent cirriform ceiling (pre-1950 data)",
           "E": "Estimated", "M": "Measured", "P": "Precipitation ceiling (pre-1950 data)", "R": "Radar",
           "S": "ASOS augmented", "U": "Unknown ceiling (pre-1950 data)", "V": "Variable ceiling (pre-1950 data)",
           "W": "Obscured", "9": "Missing"}

VQCDict = {"0": "Passed gross limits check", "1": "Passed all quality control checks", "2": "Suspect", "3": "Erroneous",
           "4": "Passed gross limits check, data originate from an NCEI data source",
           "5": "Passed all quality control checks, data originate from an NCEI data source",
           "6": "Suspect, data originate from an NCEI data source",
           "7": "Erroneous, data originate from an NCEI data source",
           "9": "Passed gross limits check if element is present"}

VQVCDict = {"0": "Passed gross limits check", "1": "Passed all quality control checks", "2": "Suspect",
            "3": "Erroneous", "4": "Passed gross limits check, data originate from an NCEI data source",
            "5": "Passed all quality control checks, data originate from an NCEI data source",
            "6": "Suspect, data originate from an NCEI data source",
            "7": "Erroneous, data originate from an NCEI data source",
            "9": "Passed gross limits check if element is present"}

TQCDict = {"0": "Passed gross limits check", "1": "Passed all quality control checks", "2": "Suspect",
           "3": "Erroneous", "4": "Passed gross limits check, data originate from an NCEI data source",
           "5": "Passed all quality control checks, data originate from an NCEI data source",
           "6": "Suspect, data originate from an NCEI data source",
           "7": "Erroneous, data originate from an NCEI data source",
           "9": "Passed gross limits check if element is present",
           "A": "Data value flagged as suspect, but accepted as a good value",
           "C": "Temperature and dew point received from Automated Weather Observing System (AWOS) are reported in whole degrees Celsius. Automated QC flags these values, but they are accepted as valid.",
           "I": "Data value not originally in data, but inserted by validator",
           "M": "Manual changes made to value based on information provided by NWS or FAA",
           "P": "Data value not originally flagged as suspect, but replaced by validator",
           "R": "Data value replaced with value computed by NCEI software",
           "U": "Data value replaced with edited value"}

SEDict = {"AU": "sourced from automated ASOS/AWOS sensors", "AW": "sourced from automated sensors",
          "MW": "sourced from manually reported present weather"}

WTDict = {"01": "Fog, ice fog or freezing fog (may include heavy fog)",
          "02": "Heavy fog or heavy freezing fog (not always distinguished from fog)",
          "03": "Thunder", "04": "Ice pellets, sleet, snow pellets or small hail",
          "05": "Hail (may include small hail)",
          "06": "Glaze or rime", "07": "Dust, volcanic ash, blowing dust, blowing sand or blowing obstruction",
          "08": "Smoke or haze", "09": "Blowing or drifting snow", "10": "Tornado, water spout or funnel cloud",
          "11": "High or damaging winds", "12": "Blowing spray", "13": "Mist", "14": "Drizzle",
          "15": "Freezing drizzle",
          "16": "Rain", "17": "Freezing rain", "18": "Snow, snow pellets, snow grains or ice crystals",
          "19": "Unknown precipitation", "21": "Ground fog", "22": "Ice fog or freezing fog"}

WCQCDict = {"0": "Passed gross limits check", "1": "Passed all quality control checks", "2": "Suspect",
            "3": "Erroneous",
            "4": "Passed gross limits check, data originate from an NCEI data source",
            "5": "Passed all quality control checks, data originate from an NCEI data source",
            "6": "Suspect, data originate from an NCEI data source",
            "7": "Erroneous, data originate from an NCEI data source",
            "M": "Manual change made to value based on information provided by NWS or FAA",
            "9": "Passed gross limits check if element is present"}

CAVOKDict = {"N": "No", "Y": "Yes", "9": "Missing"}

VVCDict = {"N": "Not variable", "V": "Variable", "9": "Missing"}
