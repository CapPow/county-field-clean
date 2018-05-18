
# coding: utf-8

# In[69]:


import pandas as pd
import numpy as np
import concurrent.futures
import re
from zipfile import ZipFile
from datetime import datetime, timedelta
from argparse import ArgumentParser  # pass in the file path


# the neame of the dwc archive to be cleaned (expected in symbiota format)
toCleanZip = 'UOS_backup_2018-05-11_163415_DwC-A.zip'

# what to place into incorrect county fields which cannot be inferred.
# Realistically, empty string is probably always the best option here.

# Load in the occurence data from the DWC archive.
parser = ArgumentParser()
parser.add_argument('-f', '--file', dest='filePath',
                    help='a zipped DWC Archive file path to be cleaned')

parser.add_argument('-c', '--problem-code', dest='problem_code', default='',
                    help='a value to be inserted when problems arise. \
                    If not specified, default is to leave it blank\
                    which is usually the best option.\
                    TAKE caution using this when not in conjunction with\
                    the -r flag.')

parser.add_argument('-r', '--review-output', dest='review',
                    action='store_true',
                    help='Instead of cleaning the records,\
                    produce a report useful for reviewing the potential\
                    changes this script would suggest.\
                    THIS IS A GOOD PLACE TO START')

args = parser.parse_args()

# extract the collection name and backup date
filePath = args.filePath
filePath = parser.parse_args().filePath
collName = str(filePath).split(r'\'')[-1].split('_backup')
collName = str(collName[0] + '_' + collName[1].split('_')[1])

with ZipFile(filePath, 'r') as dwcArchive:
    with dwcArchive.open('occurrences.csv') as data:
        data = pd.read_csv(data, dtype='str', na_values=['NAN', 'Nan', 'NaN'],
                           encoding='utf-8')

initialDataShape = data.shape  # store initial size
# limit the data to scope of cleaning (those with counties and in the US)
data.dropna(axis='index', how='any',
            subset=['county', 'country', 'stateProvince'], inplace=True)
data = data[data.country.str.contains('United States')]
print('dropped {} records which either lack county, state'
      ' or are outside the US.\nLeaving {} county fields to verify...'.format(
        initialDataShape[0] - data.shape[0], data.shape[0]))
# Load in the reference data
refDF = pd.read_csv('.//data//USA state Counties Area Reference.csv')
# address differences in reference and SERNEC's state/county associations
# These allow multiple entry options to exist, which might not be ideal.
a = refDF['State_full'] == 'FLORIDA'
b = refDF['County'].str.contains('MIAMI-DADE')
c = refDF['City_alias'] == 'MIAMI'
d = refDF['City'] == 'MIAMI'
copyRow = refDF[a & b & c & d].copy(deep=True)
copyRow['County'] = 'MIAMI DADE'
refDF = pd.concat([refDF, copyRow], axis=0, ignore_index=True)

a = refDF['State_full'] == 'MISSISSIPPI'
b = refDF['County'].str.contains('DESOTO')
copyRow = refDF[a & b][:1].copy(deep=True)
copyRow['County'] = 'DE SOTO'
copyRow['City'] = np.nan
copyRow['City_alias'] = np.nan
refDF = pd.concat([refDF, copyRow], axis=0, ignore_index=True)


# Many county errors come from the barcode scanner
# ie: dumping a catalog number after the county in the county field
# Learn catalog pattern to filter it out later
def learnPattern(data=data):
    # samples the 1 / 1000th of the dataset's catalogNumbers for equality.

    patternSet = set()
    iterationCount = 0
    while len(patternSet) != 1:  # more than 1 unique value is an error
        iterationCount += 1  # report a count of this loop's iterations.
        if iterationCount > 1:
            print('Having trouble learning the catalogNumber'
                  ' iteration count = {}').format(iterationCount)
            print('Some catalogNumbers are formatted differently,'
                  ' this is probably a mistake!'
                  '\nRegex Patterns found:{}'.format(
                          [x.pattern for x in patternSet]))
        patternSet = set()
        for i in range(int(len(data)/1000)):
            regexCatNum = data.sample(1).iloc[0]['catalogNumber']
            herbPrefix = re.findall(r'\d*\D+', regexCatNum)[0]
            numericSuffix = regexCatNum.split(herbPrefix)[-1]
            p = '(' + herbPrefix + r'\d{' + str(len(numericSuffix)) + r'})'
            # regexPattern = re.compile(p)
            regexPattern = str(p)
            patternSet.add(regexPattern)
    return min(patternSet)

regexPattern = learnPattern()  # learn the pattern


def identProblem(dfToRef, dfToClean, chunkSize=5000):
    # Returns a list of the problem catalog numbers
    probList = []

    def countyCheck(ref, row):
        state = str(row.stateProvince).upper()
        county = str(row.county).upper()
        possibleCounties = ref[ref['State_full'] == state]['County'].tolist()
        if (county in possibleCounties) or (county == ''):
            return
        else:
            probList.append(row.catalogNumber)
        return probList
    with concurrent.futures.ProcessPoolExecutor() as pool:  # speed this up.
        pool.map(dfToClean.apply(lambda row: countyCheck(dfToRef, row),
                                 axis=1), chunksize=chunkSize)
    return sorted(probList)

# warn this can be a slow process, give rudamentary time estimate.
doneEst = (datetime.now() + timedelta(
        minutes=(initialDataShape[0] / 10000) * 2)).strftime('%I:%M %p')
print('This could take until {}'.format(doneEst))

# run the identProblem returning all data fields from problematic records.
p = identProblem(refDF, data)
data = data[data['catalogNumber'].isin(p)]
print('finished at: {}'.format(datetime.now().strftime('%I:%M %p')))


# #### The large, ugly, and growing cleaning function.
# - Probably a better way to do this, but this works
# - This is the result of addressing frequent errors
#       discovered through multiple iterations
# #### potential issue:
# - data entry for Stanly county NC was "Stanley" which triggeerd a problem.
# - Turns out there is a "Stanley" municipality in Gaston county NC.
# - Therefore inappropriate inference.. not sure how to spot it.
output = {}


def cleanProblemCounty(dfToRef, dfToClean, chunkSize=5000):

    def countyCheck(countyToCheck, state):
        # is the countyToCheck is an option in the reference?
        possibleCounties = dfToRef[dfToRef['State_full'] == state]['County']
        if countyToCheck in possibleCounties:
            return True
        else:
            return False

    def countyFromCityLookUp(state, city):
        a = dfToRef['State_full'] == state
        b = dfToRef['City'] == city
        countyOptions = dfToRef[a & b]['County'].unique()
        if len(countyOptions) == 1:
            return countyOptions[0].title()
        else:
            return False

    def countyFromCityAliasLookup(state, city):
        a = dfToRef['State_full'] == state
        b = dfToRef['City_alias'] == city
        countyOptions = dfToRef[a & b]['County'].unique()
        if len(countyOptions) == 1:
            return countyOptions[0].title()
        else:
            return False

    def countyResolve(ref, row):
        # establish some local variables
        state = str(row.stateProvince).upper()

        # attempt to filter catalogNumbers from county.
        county = str(re.sub(regexPattern, '', (row.county))).strip()
        # if something changed, update county
        if row.county != county:
            row['suggested_county'] = county
        # attempt to remove any numbers that may be in the county name
        county = ''.join([i for i in county if not i.isdigit()]).upper()
        countyWordList = county.split()
        municipality = str(row.municipality).upper()
        suggestedMunicipalityName = municipality.title()
        countySuggestedFromCity = countyFromCityLookUp(state, county)
        countySuggestedFromCityAlias = countyFromCityAliasLookup(state, county)

        # make sure we did not filter the entire thing away!
        if (county == ''):
            row['suggested_county'] = args.problem_code

        # since it still exists, check it right out of the gate.
        elif countyCheck(county, state):
                suggestedCountyName = county
                row['suggested_county'] = suggestedCountyName.title()

        # if it fails, then run the gauntlet of common errors
        # does alaska even have counties? (SERNEC Accepts this one)
        elif (state == 'ALASKA') & (county == 'PRINCE WALES KETCHIKAN'):
            suggestedCountyName = 'PRINCE WALES KETCHIKAN'.title()
            row['suggested_county'] = suggestedCountyName

        # handle the De Soto "DeSoto" issue (SERNEC may have this one wrong)
        elif (state == 'MISSISSIPPI') & (county in ['DE SOTO', 'DESOTO']):
            suggestedCountyName = 'De Soto'
            row['suggested_county'] = suggestedCountyName

        # Handle (more?)  Miami - Dade issues.
        elif (state == 'FLORIDA') & (county == 'DADE'):
            suggestedCountyName = 'MIAMI DADE'.title()
            row['suggested_county'] = suggestedCountyName

        # Handle the fact that ElDorado is actually El Dorado
        elif (state == 'CALIFORNIA') & (county == 'ELDORADO'):
            suggestedCountyName = 'EL DORADO'.title()
            row['suggested_county'] = suggestedCountyName

        # I have no clue why this typo is so common but it warrents a check.
        elif (state == 'VIRGINIA') & (county == 'BOUTETOURT'):
            suggestedCountyName = 'BOTETOURT'.title()
            row['suggested_county'] = suggestedCountyName

        # People are thinking the sauce perhaps?
        elif (state == 'MASSACHUSETTS') & (county == 'WORCHESTER'):
            suggestedCountyName = 'worcester'.title()
            row['suggested_county'] = suggestedCountyName

        # handle the two extinct Virginia counties
        elif (state == 'VIRGINIA') & any(x in county for x in
                                         ['NANSEMOND', 'PRINCESS ANNE',
                                          'CITY OF VIRGINIA BEACH']):
            if county == 'NANSEMOND':
                suggestedCountyName = 'Suffolk City'
            elif county == 'PRINCESS ANNE' or 'CITY OF VIRGINIA BEACH':
                suggestedCountyName = 'Virginia Beach City'
            row['suggested_county'] = suggestedCountyName

        # handle the fact "Swain County TN" does not exist, it is NC
        elif (state == 'TENNESSEE') & (county == 'SWAIN'):
            row['stateProvince'] = 'Tennessee'
            row['suggested_county'] = 'Swain'

        # handle the fact "Sussex County NY" is actually in New Jersey
        elif (state == 'NEW YORK') & (county == 'SUSSEX'):
            row['stateProvince'] = 'New York'
            row['suggested_county'] = 'Sussex'

        # Handle the fact that dekalb is probably De kalb (in Alabama)
        elif (state == 'ALABAMA') & (county == 'DEKALB'):
            suggestedCountyName = 'DE KALB'.title()
            row['suggested_county'] = suggestedCountyName

        # Yet De kalb is actually dekalb everywhere else.
        elif (county == 'DE KALB') & (state in[
                                                'GEORGIA', 'ILLINOIS',
                                                'INDIANA', 'MISSOURI',
                                                'TENNESSEE']):
            suggestedCountyName = 'DEKALB'.title()
            row['suggested_county'] = suggestedCountyName.title()

        # try and strip the word "county" from the county name
        elif (len(countyWordList) > 1) & (countyWordList[-1] == 'COUNTY'):
                attempt = county.replace(countyWordList[-1], '').strip()
                if countyCheck(attempt, state) is True:
                    suggestedCountyName = attempt
                    row['suggested_county'] = suggestedCountyName.title()

        # "clark is not the same as clarke"
        elif county == 'CLARK':
            attempt = 'CLARKE'
            if countyCheck(attempt, state) is True:
                suggestedCountyName = attempt
                row['suggested_county'] = suggestedCountyName.title()

        # student labor comes with additional considerations
        elif county == 'COCK':
            attempt = 'COCKE'
            if countyCheck(attempt, state) is True:
                suggestedCountyName = attempt
                row['suggested_county'] = suggestedCountyName.title()

        # try to solve the "st." and/or "saint" mistakes
        elif (len(countyWordList) > 1) & (countyWordList[0] in [
                                            'ST', 'ST.', 'SAINT']):
            saintPrefixList = ['ST', 'ST.', 'SAINT']
            saintPrefixList.remove(countyWordList[0])
            for saintPrefix in saintPrefixList:
                attempt = county.replace(countyWordList[0], saintPrefix)
                if countyCheck(attempt, state) is True:
                    suggestedCountyName = attempt
                    row['suggested_county'] = suggestedCountyName.title()

        # try to resolve city names being placed into counties
        elif countySuggestedFromCity:
            suggestedCountyName = countySuggestedFromCity
            if (pd.isna(row['municipality'])) or (row[
                                                    'municipality'] is 'Nan'):
                suggestedMunicipalityName = county.title()
            row['suggested_county'] = suggestedCountyName
            row['suggested_municipality'] = suggestedMunicipalityName

        elif countySuggestedFromCityAlias:
            suggestedCountyName = countySuggestedFromCityAlias
            if (pd.isna(row['municipality'])) or (row[
                                                    'municipality'] is 'Nan'):
                suggestedMunicipalityName = county.title()
            row['suggested_county'] = suggestedCountyName
            row['suggested_municipality'] = suggestedMunicipalityName

# If all those fail, the "args.problem_code" into the field to overwrite it.

        else:
            try:
                if pd.isnull(row['suggested_county']):
                    row['suggested_county'] = args.problem_code
            except KeyError:
                row['suggested_county'] = args.problem_code

        output[row.catalogNumber] = row
    # if we titlecased any NaNs they are converted to np.nan
    dfToClean = dfToClean.replace('Nan', np.nan)
    # execute the long chain above, as a multi-threadded process.
    with concurrent.futures.ProcessPoolExecutor() as pool:
        pool.map(dfToClean.apply(lambda row: countyResolve(dfToRef, row),
                                 axis=1), chunksize=chunkSize)

    resultDF = pd.DataFrame.from_dict(output, orient='index')
    # Handle when reccomandations are made or the column is non-existant)
    for suggestedCol in ['suggested_county', 'suggested_municipality']:
        if suggestedCol not in resultDF.columns:
            resultDF[suggestedCol] = ''
    return resultDF


# actually run the cleaning function
cleanedDF = cleanProblemCounty(refDF, data).fillna('')
# Report some stats on the process
print('complete dataset = ', initialDataShape[0])
print('problem data = ', data.shape[0])
print('error rate(%) = ', ((data.shape[0]) / initialDataShape[0]) * 100)
a = cleanedDF['suggested_county'] != args.problem_code
print('suggested counties = ', cleanedDF[a].shape[0])
print('correction rate(%) = ', (cleanedDF[a].shape[0]) / (data.shape[0]) * 100)
b = cleanedDF['suggested_municipality'] != args.problem_code
c = cleanedDF['suggested_municipality'] != ''
d = cleanedDF[b & c].shape[0]
print('suggested municipalities = ', d)
print('municipality salvage rate(%) = ', (d) / (data.shape[0]) * 100)


if args.review:
    # save suggested Changes for review
    colOfInterest = ['institutionCode',
                     'scientificName',
                     'stateProvince',
                     'county',
                     'suggested_county',
                     'municipality',
                     'suggested_municipality']
    cleanedDF.to_csv('Suggested_Changes_{}.csv'.format(collName),
                     columns=colOfInterest, na_rep='',
                     encoding='utf-8', index=False)
else:
    # finalize and save suggested Changes
    cleanedDF = cleanedDF.drop(columns=['county', 'municipality'])
    cleanedDF = cleanedDF.rename(index=str,
                                 columns={
                                    'suggested_county': 'county',
                                    'suggested_municipality': 'municipality'})

    cleanedDF.to_csv('Cleaned_{}.csv'.format(collName), na_rep='',
                     encoding='utf-8', index=False)

print('Always use the most recent data\
 and upload the cleaned records promptly!')
