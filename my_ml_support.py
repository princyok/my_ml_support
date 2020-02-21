'''
@author: Prince Okoli
'''
import pandas as pd
from matplotlib import (pyplot, gridspec,
                        cm, colors, patches)
import numpy as np
import datetime
from scipy import integrate
from sklearn import (model_selection,
                     metrics,
                     preprocessing,
                     base)
import random
import re
import itertools
import functools
import pickle
import sys
import copy
pyplot.ioff()
    

class Tool():
    """
    This contains tools for for preprocessing of data and for machine learning.
    
    INSTANCE ATTRIBUTES:
        >> objTimestamp: A timestamp of when the object was created. It's a string.
        >> allClassifications: All classifications (classes and their starting
        thresholds) made by the methods called by the object. It's a list of strings.
        >> allConfusionMatrices: All confusion matrices produced by methods
        called by the object. It's a list of strings.
        >> allAccuracies: All predictive accuracies produced by methods called 
        by the object. It's a list of strings.
        >> allMethodCalls: All methods called by the object that are also decorated by the 
        decorator-function _recordFunctionCall(). It's a list of strings.        
    """
    def __init__(self):
                # all instance attrs are strings or list of strings.
        self.objTimestamp=self._generateTimestamp()
        self.allClassifications=None 
        self.allConfusionMatrices=None 
        self.allAccuracies=None
        self.allAveragePrecisions=None
        self.allMethodCalls=None
        
    def _generateTimestamp(self):
        currentTime=datetime.datetime.now().timestamp()
        timestamp=datetime.datetime.fromtimestamp(currentTime).isoformat()
        return timestamp
        
    def _updateAllClassifications(self, usedClassesAndThresholds):
        if (self.allClassifications==None): self.allClassifications=list()
        self.allClassifications.append(str(usedClassesAndThresholds))
        return None
    
    def _updateAllMethodCalls(self, mthdCallLiteral):
        if (self.allMethodCalls==None): self.allMethodCalls=list()
        self.allMethodCalls.append(mthdCallLiteral)
        return None
    
    def _updateAllConfusionMatrices(self, confusionMatrix):
        if (self.allConfusionMatrices==None): self.allConfusionMatrices=list()
        if isinstance(confusionMatrix, np.ndarray): confusionMatrix=np.array2string(confusionMatrix)
        self.allConfusionMatrices.append(confusionMatrix)
        return None    
    
    def _updateAllAccuracies(self, accuracy):
        if (self.allAccuracies==None): self.allAccuracies=list()
        self.allAccuracies.append(str(accuracy))
        return None
    
    def _updateAllAveragePrecisions(self, avgPrecision):
        if (self.allAveragePrecisions==None): self.allAveragePrecisions=list()
        self.allAveragePrecisions.append(str(avgPrecision))
        return None
    
    def _getAllObjAttrNames(self):
        allObjAttrNames=list(self.__dict__.keys())
        return allObjAttrNames
    
    def _isIterable(self, arg):
        """
        Checks if an object or data is iterable, and returns True if it is.
        """
        try:
            (i for i in arg)
            ans=True
        except TypeError:
            ans=False
        return ans
    
    def _multi_replace(self, string, repMap):
        """
        Replace multiple substrings in a string with the same or different replacements.
        
        PARAMETERs:
            **string**: string that contains the substrings to be replaced.
            **repMap**: a dict of the substrings (key) and replacements (val).
        """
        substring = sorted(repMap, key=len, reverse=True)
        regexp = re.compile('|'.join(map(re.escape, substring)))
        return regexp.sub(lambda match: repMap[match.group(0)], string)
    
    def generateLog(self):
        """
        OPs:
            Generates a log of all the data in the instance attributes.               
        RETURN:
            Returns a log. The log is a dictionary where the key is the name of
            an instance attribute and the value is the data held by the instance attribute.
        """
        allObjAttrNames=self._getAllObjAttrNames()
        allObjAttrs=list()
        
        for AttrName in allObjAttrNames:
            Attr=getattr(self, AttrName)
            if (Attr==None): Attr=list()
            allObjAttrs.append(Attr)
            
        log=dict(zip(allObjAttrNames, allObjAttrs))
        log["timestamp_of_log"]=self._generateTimestamp()
        return log
    
    def writeLogToFile(self, filepath, mode="a"):
        """
        OPs:
            writes log to file.
        """
        log=self.generateLog()
        logDF = pd.DataFrame({key:pd.Series(value) for (key, value) in log.items()})
        logDF.to_csv(filepath, encoding="utf-8",index=False, mode=mode)               

        return None
    
    def _formatArgValuePair(self, arg_val):
        """ 
        ARGs, OPs and RETURN:
            Accepts a tuple of the form (name, value) and returns the string "name=value".
        EXAMPLE:
        In: _formatArgValuePair(('x', (1, 2, 3)))
        Out: 'x=(1, 2, 3)'
        """
        arg, val = arg_val
        return "%s=%r" % (arg, val)
        
    def _recordFunctionCall(method):
        """
        ARGs:
            Accepts a function object.
        OPs and RETURN:
            This decorator returns a function object that wraps the argument function, 
            producing a traced version of that function. 
            This decorator decorates other methods, thereby logging each of their calls.
            The methods typically decorated by this decorator are the non-private 
            instance methods.
        """
        
        # Get the function's arg count, arg names, arg defaults.
        code = method.__code__
        argCount = code.co_argcount # number of args excluding flexible args.
        argNames = code.co_varnames[:argCount] # names of args excluding flexible args.
        mthdDefaults = method.__defaults__ if (method.__defaults__) else list()
        dictOfDefaults = dict(zip(argNames[-len(mthdDefaults):], mthdDefaults))
        
        @functools.wraps(method)
        def wrapper(*pargs, **kwargs):
            dtypesToExcludeDetialsFor=(pd.DataFrame, pd.Series)
            # set the representation of pandas objs to the representation of their types.
            p=tuple()
            for arg in pargs:
                if isinstance(arg, dtypesToExcludeDetialsFor):
                    p+=((type(arg)),)
                else:
                    p+=(arg,)
            kw=dict()
            for k in kwargs:
                if isinstance(kwargs[k], dtypesToExcludeDetialsFor):
                    kw.update({k:type(kwargs[k])})
                else:
                    kw.update({k:kwargs[k]})
                    
            # Collect the function arguments: positional, default, flexible positional, 
            # and keyword arguments (both flex and non-flex).
            allPositionals = list(map(pargs[0]._formatArgValuePair, zip(argNames, p)))
            allDefaultsExecuted = [pargs[0]._formatArgValuePair((a, dictOfDefaults[a]))
                         for a in argNames[len(p):] if a not in kw]
            AllFlexiblePargs = list(map(repr, p[argCount:]))
            AllKwargs = list(map(pargs[0]._formatArgValuePair, kw.items()))
            allArgs = allPositionals + allDefaultsExecuted + AllFlexiblePargs + AllKwargs # adding the lists.
            mthdCallLiteral=("%s(%s)" % (method.__name__, ", ".join(allArgs)))
            pargs[0]._updateAllMethodCalls(mthdCallLiteral)
            return method(*pargs, **kwargs)
        
        return wrapper

    def _getFileType(self, filepath):
        """
        OPs:
            Extracts file format from a filename, and return the file format (without the period).
            The file format is the substring after that last period in a string. 
            If there is no file format, None is returned.
        """
        fileFormat=filepath.split(".")[-1]
        if fileFormat==filepath: fileFormat=None
        return fileFormat
    
    @_recordFunctionCall
    def _getFileTypes(self, filepaths):
        """
        OPs:
            Extract file formats from a list of filenames, and return as a 
            list in an order matching the argument.
        """
        allFileFormats=[]
        for filepath in filepaths:
            fileFormat=self._getFileType(filepath)
            allFileFormats.append(fileFormat)
        return allFileFormats
    
    def _verifySameFileFormats(self, filepaths):
        '''
        OPERATION: takes a list of filepaths and verifies whether they are all the same format.
        If yes, it returns True, else False.
        '''
        all_FileFormats=self._getFileTypes(filepaths)
        # returns True if all file formats are the same, and False if not.
        all_fileFormats_are_same=all((fileFormat==all_FileFormats[0] for fileFormat in all_FileFormats))
        return all_fileFormats_are_same
    
    @_recordFunctionCall
    def getData(self, filepaths, pandasReaders={"csv":pd.read_csv,"xlsx":pd.read_excel}):
    
        listOfDataFrames=[]
        for filepath in filepaths:
            fileFormat=self._getFileType(filepath)
            try:
                df=pandasReaders[fileFormat](filepath)
            except KeyError:
                sys.exit("The file cannot be opened. Pass the appropriate"+ 
                         "pandas reader to the pandasReaders argument.")
            listOfDataFrames.append(df)
        
        if not all((np.array_equal(dFrame.columns,listOfDataFrames[0].columns) for dFrame in listOfDataFrames)):
            sys.exit("Two or more of the datasets have non-matching column labels.")
            
        wholeData=pd.concat(listOfDataFrames, ignore_index=True)
        return wholeData
    
    @_recordFunctionCall
    def generateRandomImitation(self, df, colNamesToReplicateExactly=["Time"], maxRangeInStd=3):
        dictOfCols=dict()
        for colName in df.columns:
            dictOfCols.update({colName:[]})
    
        for c in colNamesToReplicateExactly:
            del dictOfCols[c]
    
        lenDF=len(df.index)
    
        for colName in dictOfCols:
            colData=df[colName]
    
            lowestPossibleLim=colData.mean()-(maxRangeInStd*(colData.std())) # default is 3 std below mean.
            highestPossibleLim=colData.mean()+(maxRangeInStd*(colData.std())) # default is 3 above mean.
    
            if colData.min()<lowestPossibleLim:
                lowerLim=lowestPossibleLim
            else:
                lowerLim=colData.min()    
    
            if colData.max()>highestPossibleLim:
                higherLim=highestPossibleLim
            else:
                higherLim=colData.max()
    
            for _ in range(lenDF):
                dictOfCols[colName].append(random.SystemRandom().uniform(lowerLim,higherLim))
    
        randomDF=pd.DataFrame(dictOfCols)
    
        for c in colNamesToReplicateExactly:
            randomDF[c]=df[c].reset_index(drop=True)
    
        return randomDF
    
    @_recordFunctionCall
    def covertTimeToDateTime(self, df,colName):
        df=df.copy()
        df.loc[:, colName]=pd.to_datetime(df[colName])
        return df
    		
    @_recordFunctionCall
    def renameColumns(self, df, mapForRenaming):
        df=df.copy()
        colNames=df.columns
        for pattern in mapForRenaming:
            for colName in colNames:
                if re.search(pattern,colName):
                    mapForRenaming[colName] = mapForRenaming.pop(pattern)
    
        renamedDF=df.rename(columns=mapForRenaming)    
        return renamedDF
    
    @_recordFunctionCall
    def simplifyColumnNamesWithUnderscore(self, df, to_replace):
        """
        Replace all specified substrings with the specified replacements in all columns names.
        PARAMETERs:
            to_replace: list of substrings (or a string) to replace with underscore.
        """
        if isinstance(to_replace,str): to_replace=[to_replace]
        
        df=df.copy()
        
        repMap=dict(zip(to_replace,["_"]*len(to_replace)))        
                    
        return self.simplifyColumnNames(df, repMap=repMap)
    
    @_recordFunctionCall
    def simplifyColumnNames(self, df, repMap):
        colNames=df.columns.to_numpy()
        for i,col in enumerate(colNames):
            colNames[i]=self._multi_replace(string=col.strip(), repMap=repMap)
        return df
    
    @_recordFunctionCall
    def getDataAboveThreshold(self, df, colName, threshold):
        data=df[df[colName]>threshold]
        return data
    
    @_recordFunctionCall
    def getPortion(self, df, proportion, start_from='top'):
        """
        ARGs:
            >> df: the dataset (as a DataFrame).
            >> proportion: the portion of the dataset to get. A number between
             0 and 1 (endpoints inclusive), where 1 is the entire dataset and 
             0 is nothing.
            >> start_from: takes one of the three strings, "top", "bottom", and
             "anywhere", that specifies where to start from, and the strings are
             not case-sensitive. If "anywhere", the proportion must be an 
             array-like in the form [start, end].
            
        OPs:
            Extract a portion of a dataset.
        """
        ind=df.index.get_values()
        lenDF=len(ind)   
        portion=None
        if start_from.lower() == 'top':
            indToDrop=ind[int(round(lenDF*proportion)):lenDF]
            portion=df.drop(indToDrop).copy()
        elif start_from.lower() == 'bottom':
            indToDrop=ind[0:int(round(lenDF*(1-proportion)))]
            portion=df.drop(indToDrop).copy()
        elif start_from.lower() == 'anywhere':
            proportion=np.array(proportion)
            try:
                if len(proportion)<2:
                    sys.exit("proportion: valid type")
            except TypeError:
                sys.exit("proportion: valid type")
            indToGet=ind[int(round(lenDF*proportion[0])):int(round(lenDF*proportion[1]))]
            portion=df.loc[indToGet,:].copy()
        else:
            print("start_from: Invalid string.")
        return portion
    
    @_recordFunctionCall
    def generateResultantMagnitude(self, df, colNamesComponents=None, resultantColName=""):
        df=df.copy()
        if (colNamesComponents==None): colNamesComponents=[""]
        interim=0
        for comp in colNamesComponents:
            interim+=df[comp].pow(2)
        df.loc[:, resultantColName]=np.sqrt(interim)
        return df
    
    @_recordFunctionCall
    def sma(self, df,sma_period,sma_variables,suffix="_SMA"):
        df=df.copy()                           
#        # check if there is any NaN, and fills them using the specified approach.
#        
#        if len(sma_variables)==1:
#            # if only one variable.
#            anyNan=df.loc[:,sma_variables].isna().any()
#        else:
#            # if more than one variable.
#            anyNan=df.loc[:,sma_variables].isna().any().any()
#        
#        if anyNan:
#            if fillNaN=="ffill" or fillNaN=="pad" or fillNaN=="forwardfill":
#                ## first foward-fill all NaNs.
#                smaDF=df.loc[:,sma_variables].fillna(method="ffill")
#                ## then do a backward-fill incase first value is NaN.
#                smaDF.fillna(method="bfill",inplace=True)
#            
#            elif fillNaN=="bfill" or fillNaN=="backfill":
#                ## first back-fill all NaNs.
#                smaDF=df.loc[:,sma_variables].fillna(method="bfill")
#                ## then do a forward-fill incase last value is NaN.
#                smaDF.fillna(method="ffill",inplace=True)
#        else:
#            smaDF=df.loc[:,sma_variables]
        
        for sma_variable in sma_variables:
            
            weight = np.repeat((1 / sma_period),sma_period) 
            sma = np.convolve(df.loc[:,sma_variable], weight,'valid') # using convolution to compute SMA
    
            #fill SMA data with NaN to match size of dataset
            nanVector = ([float('NaN')]*(sma_period - 1))
            nanVector.extend(sma)
            df.loc[:,sma_variable+suffix]=nanVector
        
        # drop the NaNs from the SMA columns.
        for sma_variable in sma_variables:   
            df=df.drop(index=df.loc[df.loc[:,sma_variable+suffix].isna(),:].index)
        return df
    
    @_recordFunctionCall
    def convertDegreeToRadian(self, df,colName):
        df=df.copy()
        df[colName]=(df[colName]*np.pi)/180
        return df
    
    @_recordFunctionCall
    def computeVibrationalMomentum(self, df, colName="",newColName="",timeInterval=0.5,initialVal=0):
        df=df.copy()
        df[newColName]=integrate.cumtrapz(df[colName],dx=timeInterval,initial=initialVal)
        return df
    
    @_recordFunctionCall		
    def zScoreNormalizeDF(self, df, colNamesToExclude=None):
        '''
        ARGs:
            >> df: the dataset that is to be normalized. A pandas DataFrame.
            >> colNamesToExclude: colunm labels of columns that should be excluded 
            from the normalization.
        OPs:
            Normalizes a dataset via the z-score (standard score) normalization. 
            Computes the mean and standard dev using the dataset. 
        '''
        df=df.copy()
        if (colNamesToExclude==None): colNamesToExclude=[""]
        lenCols=len(df.columns)
        normalizedDF=pd.DataFrame()
    
        excludedColumnIndices=[]
        for excludedColumnName in colNamesToExclude:
            excludedColumnIndex=df.columns.get_loc(excludedColumnName)
            excludedColumnIndices.append(excludedColumnIndex)
    
        for i in range(0,lenCols):
            colName=df.columns[i]
            if not i in excludedColumnIndices:
                normalizedDF[colName]=(df[colName] - df[colName].mean())/ df[colName].std()
            else:
                normalizedDF[colName]=df[colName]    
        return normalizedDF
    
    @_recordFunctionCall
    def zScoreNormalizeNum(self, num, unNormalizedData):
        '''
        ARGs:
            >> num: the number that is to be normalized. Numerical type.
            >> unNormalizedData: the unnormalized dataset from which the number 
            that's to be normalized origniates from. A pandas Series.
        OPs:
            Normalizes a number via the z-score (standard score) normalization. 
            Computes the mean and standard dev using the unnormalized dataset. 
        '''
        normalizedNum=(num - unNormalizedData.mean())/ unNormalizedData.std()
        return normalizedNum
    
    def computeGradientViaCentralDiff(df, colName, stepsize, suffix="_Gradient"):
        df=df.copy()
        newColName=colName+suffix
        #TODO verify this formula is correct.
        df.loc[:,newColName]=(df.loc[:,colName].shift(-stepsize)-df.loc[:,colName].shift(stepsize))/(2*stepsize)
        df.dropna(inplace=True)
        return df

    def minmaxNormalize(self, df, customMinMax=None):
        """
        Scales each variable in the data such that the minimum and maximum values 
        are rescaled to 0 and 1, and values that are inbetween are scaled proportionally.
        """
        df=df.copy()
        if customMinMax==None:
            df=(df-df.min())/(df.max()-df.min())
        else:
            for colName in customMinMax:
                min_, max_ = customMinMax.get(colName)
                df.loc[:,colName]=(df.loc[:,colName]-min_)/(max_ - min_)
        return df
    
    @_recordFunctionCall
    def classifyAccordingToThresholds(self, df, colNameToClassify, thresholds, classColName="Classes",classPrefix=None):
        '''
        ARGs:
            
            >> df: a DataFrame that contains the dataset that is to be classified 
            into classes.
            
            >> colNameToClassify: a string that represents the column name of 
            the column in the dataset that is to be classified/binned.
            
            >> thresholds: an iterable of numbers (list, numpy array, etc.) that 
            represents the starting thresholds for intervals (i.e. the classes). 
            The mimimum number in the dataset that is to be classified is automatically
            the first starting threshold (i.e. for "class1"), and therefore should 
            not be part of this argument. The thresholds are administered in order,
            e.g. the second interval (which corresponds with the first starting
            threshold in this argument) will be named "class2" if the classPrefix
            is "class", and third interval will be named "class3", and so on.
            
            >> classColName: the name of the column that will contain the classes.
            
            >> classPrefix: a string representing the prefix for the classes. 
            Suffices (integers as strings, starting from 1) are automatically appended. 
            Default is None, which results in ordinal encoding.
            
        OPs and RETURN:
            
            Generates bins for a given variable based on the specified thresholds. 
            It returns a DataFrame that has a column of the classification named "Classes",
            and a list of tuples, where the tuples are in the form of (class, startThresh).
        '''
        df=df.copy()
        if classPrefix==None:
            classes=[str(n) for n in range(1,len(thresholds)+2)]
            classPrefix=""
        else:
            classes=[classPrefix+str(n) for n in range(1,len(thresholds)+2)]
        
        classesAndThresholds=list(zip(classes[1:],thresholds))
        classesAndThresholds=[(classes[0],df[colNameToClassify].min())]+classesAndThresholds
    
        df[classColName]=classes[0]
        for (cls,startThresh) in classesAndThresholds:
            df[classColName]=df[classColName].mask(df[colNameToClassify]>=startThresh, cls)
    
        usedClasses=list(df[classColName].unique())
        
        usedClasses=sorted(usedClasses,key=lambda k: int(k[len(classPrefix):]))
    
        usedClassesAndThresholds=[]
        for usedClass in usedClasses:
            for classAndLevel in classesAndThresholds:
                if usedClass==classAndLevel[0]:
                    usedClassesAndThresholds.append((usedClass,classAndLevel[1]))
                    
        self._updateAllClassifications(usedClassesAndThresholds)
        return (df, usedClassesAndThresholds)
    
    @_recordFunctionCall
    def classifyEvenly(self, df, colNameToClassify, numOfClasses=5, classColName="Classes",classPrefix=None):
        '''
        ARGs:
            >> df: DataFrame. The dataset that is to be classified into classes.
            >> colNameToClassify: String. The name of the column in the dataset
            that will be classified/binned.
            >> numOfClasses: Integer. The number of classes that the variable will be evenly classed into.
            >> classColName: String. The name of the column that will contain the classes.
            >> classPrefix: String. The prefix for the classes. The suffices (which are automatic) 
            are integers, starting from 1.
        OPs and RETURN:
            Generates equally sized-bins for a given variable in a dataset. 
            The classification goes from smallest to biggest.
            It returns a DataFrame that has a column of the classification named "Classes",
            and a list of tuples, where the tuples are in the form of (class, startThresh).
        '''
        df=df.copy()
        lenDF=len(df.index)
        numPerClass=lenDF//numOfClasses
        locOfLevels=[]
        locOfLevel=0
        for i in range(1, numOfClasses): # loops one time less than the number of classes.
            locOfLevel+=numPerClass
            locOfLevels.append(locOfLevel)
            
        thresholds=list(df.sort_values(by=[colNameToClassify], inplace= False)[colNameToClassify].
                        iloc[locOfLevels].values)
        
        (df, usedClassesAndThresholds) = self.classifyAccordingToThresholds(df=df, 
                                            colNameToClassify=colNameToClassify, 
                                            thresholds=thresholds, 
                                            classColName=classColName, 
                                            classPrefix=classPrefix)
        return (df, usedClassesAndThresholds)    
    
    @_recordFunctionCall
    def resampleToBalance(self, df, classColName):
   
        df=df.copy()
        
        # get number of instances in the smallest class (by instances).
        numberOfInstances=df.groupby(by=classColName).size().min()
        
        classes=df.loc[:,classColName].unique()
        result=pd.DataFrame(columns=df.columns)
        origDtypes=df.dtypes
        
        for cls in classes:
            selection=df[df.loc[:,classColName]==cls].iloc[0:numberOfInstances].copy()
            
            # reset index and drop it into the df as a new col.
            selection.reset_index(inplace=True)
            
            result=result.append(selection, ignore_index=True,sort=False)
        # ensure that the dropped index col is integer.    
        result.loc[:,"index"]=result.loc[:,"index"].astype('int32')
        
        # set the dropped index col as the new index.
        result.set_index(keys="index", drop=True, inplace=True)
        
        # set the dtypes back to those in the original df.
        for colName,dtype in origDtypes.to_dict().items():
            result.loc[:,colName]=result.loc[:,colName].astype(dtype)
        
        return result.sort_index(axis=0) # restore original order, in case if originally sequential.

        
    @_recordFunctionCall
    def applyRollingWindow(self, df, featuresNames, winSize, special=None):
        '''
        ARGs:
            df: DataFrame.
            featuresNames: list of Strings. Represent the column labels of the features that will be rolled,
            winSize: Integer. Represents the window size of the rolling window.
            special: String. Represents special instruction for the operation.
        OPs and RETURN: 
            Generates a rolling window of a dataset.
            It returns a DataFrame that contains the additional columns produced by the rolling, 
            and a list of strings that represent the column labels of the features that were generated by the rolling operation.
        '''    
        df=df.copy()
        all_RW_Features=[]
    
        for rw_Feature in featuresNames:
    
            for i in range(1,winSize+1):
                name=rw_Feature+str(i)
                all_RW_Features.append(name) #for easy access later
                df.loc[:,name]=df.loc[:,rw_Feature]
                df.loc[:,name]=df.loc[:,name].shift(i)
                
                if special=="Derivative":
                    df.loc[:,name]=(df.loc[:,rw_Feature]-df.loc[:,name])/i
    
        # drop the first few rows. Amount to be dropped is equal to window size.
        df=df.drop(df.index[:winSize])
        return (df, all_RW_Features)
    
    def getLocsOfContiguousSpan(self, df, classColName, fillNaN=None):
        '''
        PARAMETERs:
            **df**: DataFrame of the dataset.
            **classColName**: String. The name of the class column
            **fillNaN**: String or NoneType. Specifies how to handle missing values (NaNs). Options are:
                "ffill", "pad" or "forwardfill": propagate last valid observation forward to fill NaN.
                "backfill" or "bfill": use next valid observation to fill NaN.
                None: NaN is replaced with the string "Not_a_Number".

        RETURNs:
            a dictionary where each key is a class, and the value is a list of
            the tuples of the form (start, end), where start and end are the row 
            indices for the beginining and end of a contiguous span of the class, and the 
            list of tuples contains the locations for all the contiguous spans of the class.
        '''

        # reduces the df to a single-col df containing just the col of interest.
        df=df.loc[:, classColName].copy().to_frame()

        # check if there is any NaN, and fills them using the specified approach.
        
        if df.isna().any().any():
            if fillNaN=="ffill" or fillNaN=="pad" or fillNaN=="forwardfill":
                ## first foward-fill all NaNs.
                df.fillna(method="ffill",inplace=True)
                ## then do a backward-fill incase first value is NaN.
                df.fillna(method="bfill",inplace=True)
            elif fillNaN=="bfill" or fillNaN=="backfill":
                ## first back-fill all NaNs.
                df.fillna(method="bfill",inplace=True)
                ## then do a forward-fill incase last value is NaN.
                df.fillna(method="ffill",inplace=True)
            elif fillNaN==None:
                df.fillna(value="Unknown", inplace=True)
        
        classes=df.loc[:,classColName].unique().tolist()
        tempClasses=list(range(0,len(classes)))
        
        # replace classes with temporary new classes (just numbers).
        
        replacementMap=dict(zip(classes,tempClasses))

        df=df.replace(to_replace=replacementMap, value=None)
        
        # get all the start and end locations for every contiguous span of a class, for every class.
        
        df.loc[:,"StartIdentifier"]=df.loc[:,classColName]-df.loc[:,classColName].shift(periods=1)
        df.loc[:,"EndIdentifier"]=df.loc[:,classColName]-df.loc[:,classColName].shift(periods=-1)
        classes_StartAndEndLocs=dict()
        for tempCls in tempClasses:
            startLocs=df.loc[(df[classColName]==tempCls) & (df["StartIdentifier"]!=0),"StartIdentifier"].index
            endLocs=df.loc[(df[classColName]==tempCls) & (df["EndIdentifier"]!=0),"EndIdentifier"].index
            startAndEndLocs=list(zip(startLocs.tolist(),endLocs.tolist()))
            className = [k for k, v in replacementMap.items() if v == tempCls][0]
            classes_StartAndEndLocs[className]=startAndEndLocs
        
        return classes_StartAndEndLocs
    
    def oneHotEncode(self, df, colNames, drop=False):
        """
        Applies onehot encoding to specified columns.
        
        Parameters:
            
            df: the DataFrame.
            
            colNames: the colunm (str or array-like) or columns (array-like) to encode.
            
        Returns:
            a DataFrame with the onehot-encoded data (original columns are excluded).
        """
        if isinstance(colNames,str): colNames=[colNames]
        colNames=np.array(colNames) # covert to np array in case it already isn't.
        df=df.copy()
        
        # identify all the binary categorical columns (col with 2 or less unique values).
        colNames_ordinal_encode=[]
        locs=[]
        for i,colName in enumerate(colNames):
            cats=df.loc[:,colName].unique()
            if len(cats)<=2:
                colNames_ordinal_encode.append(colName)
                locs.append(i)   
        
        # apply ordinal encoding to binary categorical columns.
        # Same result as onehote encoding, without generatin new columns.
        if len(colNames_ordinal_encode)>0:
            
            colNames=np.delete(colNames,locs) # remove the already encoded binary categorical columns.
            
            encoder=preprocessing.OrdinalEncoder(dtype=np.int32)
            
            # convert to np array and ensure its 2D.
            temp=df.loc[:,colNames_ordinal_encode].to_numpy().reshape(-1,len(colNames_ordinal_encode))
            df.loc[:,colNames_ordinal_encode]=encoder.fit_transform(temp)
        
            # ensure dtypes of the encoded columns are int.
            for c in colNames_ordinal_encode:
                # for some reason, doing it all at once for all columns doesn't do anything.
                df.loc[:,c]=df.loc[:,c].astype("int32") 
        
        # apply onehot-encoding to the rest.
        encoder=preprocessing.OneHotEncoder(dtype=np.int32)
        encodedArr=encoder.fit_transform(df.loc[:,colNames].to_numpy()) # returns a scipy csr_matrix object.
        encodedArr=encodedArr.toarray() # convert to np array.
        
        newColNames=encoder.get_feature_names(input_features=colNames) # returns an np array of lists, each with one elem (the column name).
        newColNames=newColNames.ravel() # unpack the np array to just an array of column names.
        
        # combine the newly onehot-encoded array to the df.
        df=df.join(pd.DataFrame(encodedArr, columns=newColNames, index=df.index))
        
        # omit the old columns.
        colNames_to_retain=list(colNames_ordinal_encode)+list(newColNames)
        df=df.loc[:,colNames_to_retain]
        
        return df
    
    def _normalizeMatrix(self, matrix):
        normMatrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
        return normMatrix
    
    def _normalizeVector(self, vector):
        normVector = vector.astype('float') / vector.sum(axis=0)[:, np.newaxis]
        return normVector		
    
#    @_recordFunctionCall
#    def calculateAveragePrecision(self, precision, averageType="arithmetic"):
#        if(averageType=="arithmetic"):
#            p=np.array(precision)
#            avgP=p.mean()
#        
#        self._updateAllAveragePrecisions(avgP)
#        return avgP
    
    @_recordFunctionCall
    def train(self, df, featureColLabels, targetColLabel, model, train_size=0.5, 
              random_state=2,shuffle=False, return_all=False):
        
        fts = df[featureColLabels]
        features = fts.values # converts the DataFrame obj to a 2-D numpy array.
        
        tgs = df[targetColLabel]
        target = tgs.values 
        
        (features_train, features_test, target_train, target_test) = model_selection.train_test_split(
        features, target, train_size=train_size, random_state=random_state,shuffle=shuffle)
        
        trainedModel=model.fit(features_train, target_train)
        if return_all is True:
            return trainedModel, features_train, features_test, target_train, target_test
        elif return_all is False:
            return trainedModel, features_test, target_test
        else:
            return None
    
    @_recordFunctionCall
    def test(self, features_test, target_test, trainedModel, normalize_cm=True):
        predictions = trainedModel.predict(features_test)
        accuracy=metrics.accuracy_score(target_test, predictions)
        confusionMatrix=metrics.confusion_matrix(target_test, predictions)
        precision=metrics.precision_score(target_test,predictions, average=None)
        
        if normalize_cm==True:
            # normalize the confusion matrix.
            confusionMatrix=self._normalizeMatrix(confusionMatrix)
        
        self._updateAllConfusionMatrices(confusionMatrix)
        self._updateAllAccuracies(accuracy)
        return (accuracy, precision, confusionMatrix, predictions)
    
    @_recordFunctionCall
    def trainAndTest(self, df, featureColLabels, targetColLabel, model, train_size=0.5, 
                     random_state=2,shuffle=False, normalize_cm=True, return_all=False):
        '''
        ARGs:
            >> df: a pandas DataFrame of the dataset.
            >> featureColLabels: a list of strings that represents the colunm name of the features.
            >> targetColLabel: a string that represents the colunm name of the target.
            >> model: an sklearn model constructor.
        OPs:
            Trains and tests the dataset using the specified model.
        RETURN:
            >> If return_all is not a boolean: it returns None.
            >> If return_all is False: It returns the trained model (sklearn model), 
            the predictive accuracy of the test (numerical type),
            and the confusion matrix (numpy ndarray).
            >> If return_all is True: It returns the trained model (sklearn model), 
            the features training and test datasets (both numpy ndarrays), 
            the target training and test datasets (both numpy ndarrays),
            the predictive accuracy of the test (numerical type), and the 
            confusion matrix (numpy ndarray).
        '''
        trainedModel, features_train, features_test, target_train, target_test=self.train(df, featureColLabels=featureColLabels,
                                                        targetColLabel=targetColLabel,
                                                        model=model, train_size=train_size, 
                                                        random_state=random_state, shuffle=shuffle, return_all=True)
    
        (accuracy, precision, confusionMatrix)=self.test(features_test, target_test, trainedModel, normalize_cm=normalize_cm)
        
        if return_all is True:
            result = (trainedModel, features_train, features_test, target_train, 
                      target_test, accuracy, precision, confusionMatrix)
        elif return_all is False:
            result = (trainedModel, accuracy, precision, confusionMatrix)
        else:
            result = None
        return result
    
    def loadSavedModel(self, filepath):
        with open(filepath, 'rb') as fileHandle:
            model=pickle.load(fileHandle)
        return model
    
    def saveTrainedModel(self, model, filepath):
        with open(filepath, 'wb') as fileHandle:
            pickle.dump(model, fileHandle)
        return None
    
    @_recordFunctionCall
    def TimeSeriesCrossValidate(self, df, featureColLabels, targetColLabel, model, 
                           n_splits, scorings='accuracy'):
        splittingStrategy=model_selection.TimeSeriesSplit(n_splits=n_splits)
        result=self._crossValidate(df=df,featureColLabels=featureColLabels, targetColLabel=targetColLabel,
                            model=model, splittingStrategy=splittingStrategy, scorings=scorings)
        return result

    @_recordFunctionCall    
    def _crossValidate(self, df, featureColLabels, targetColLabel, model, splittingStrategy, scorings='accuracy'):
        # this method only works for multiclass when the default of the parameter "average"
        # in the signature of precision_score(...) is set to an option that's compatible with multiclass.
        # The method can be found in "lib/site-packages/sklearn/metrics/classification.py".
        fts = df[featureColLabels]
        features = fts.values # converts the DataFrame obj to a 2-D numpy array.
        
        tgs = df[targetColLabel]
        target = tgs.values
        
        kfold = splittingStrategy
        
        scores=list()
        for scoring in ([scorings] if isinstance(scorings, str) else scorings):
            # if scoring is a string put it into a list, else iterate through scorings as is.
            scores += [model_selection.cross_val_score(model, features, target, cv=kfold, scoring=scoring)]
            model=base.clone(model) # reset the model.
            
            if scoring=="precision":
                self._updateAllAveragePrecisions(scores[-1])
                
        return (scores if len(scores)!=1 else scores[0])
    
    def _addTimestampToFilename(self, filepath):
        timestamp=self._generateTimestamp().replace(":","_")
        
        fileFormat=self._getFileType(filepath)
        if fileFormat==None: raise Exception("No file format was detected. Check the filepath.")
        
        stampedFilepath=filepath.replace("."+fileFormat,"")+timestamp+"."+fileFormat
        
        return stampedFilepath                            
    
    @_recordFunctionCall
    def _drawVector(self, vector, classes, title='Precision', 
                            cmap=None, figsize=None, ylabel='True classes'):
        if cmap is None: cmap=pyplot.cm.Blues
        if figsize is None: figsize=[7,5]
        
        vector=vector*100
        # Bending array to the right shape (i.e. to 2D).
        vector=np.atleast_2d(vector).transpose()
        
        fig=pyplot.figure(figsize=figsize)
        ax = fig.add_subplot(1,1,1)
        
        axImg=ax.imshow(vector, cmap=cmap)
        fig.colorbar(axImg, fraction=0.046, pad=0.04)
        
        ax.set_title(title)
        
        tick_positions = np.arange(len(classes))
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(classes)
        # remove any x-axis tick marks, labels, etc.
        ax.tick_params(axis='x',which='both', bottom=False, top=False, labelbottom=False) 
        
        fmt = '.2f'
        threshold = (np.nanmax(vector)+np.nanmin(vector)) / 1.5
        r=vector.shape[0]
        c=vector.shape[1]
    
        for i, j in itertools.product(range(r), range(c)):
            ax.text(j, i, format(vector[i, j], fmt)+"%",
                     horizontalalignment="center",
                     color="white" if (vector[i, j] > threshold) else "black")
    
        ax.set_ylabel(ylabel)
        ax.set_xlabel('')
                
        return fig.axes
        
    @_recordFunctionCall
    def plotVector(self, vector, classes, title='Precision', 
                            cmap=None, figsize=None, ylabel='True classes', save=False, 
                            filepath="precision.png", timestampFilename=True):
        
        axes=self._drawVector(vector, classes, title=title, 
                            cmap=cmap, figsize=figsize, ylabel=ylabel)
        
        fileFormat=self._getFileType(filepath)
        if fileFormat==None: raise Exception("No file format was detected. Check the filepath.")
        
        ax=axes[0]        
        fig=ax.get_figure()
        
        # add colorbar.
#        axImg=ax.get_images()[0]
#        fig.colorbar(axImg)
        
        fig.tight_layout()
        
        if timestampFilename is True:
            filepath=self._addTimestampToFilename(filepath)

        if save is True: fig.savefig(filepath)
        
        fig.show()
        
        return None
    
    def _drawMatrix(self, matrix, classes, title='Confusion matrix', 
                            cmap=None, figsize=None, ylabel='True classes', xlabel='Predicted classes'):
        
        if cmap is None: cmap=pyplot.cm.Reds
        if figsize is None: figsize=[7,5]
        
        if not ((matrix.sum(axis=1)<=1).all()==True):
			# a test dataset of only 1 datapoint will fool the above condition.
            matrix=self._normalizeMatrix(matrix)
        
        matrix=matrix*100
        
        fig=pyplot.figure(figsize=figsize)
        ax=fig.add_subplot(1,1,1)
        
        axImg=ax.imshow(matrix, cmap=cmap)
        fig.colorbar(axImg, fraction=0.046, pad=0.04)
        
        ax.set_title(title)
        
        tick_positions = np.arange(len(classes))
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(classes, rotation=45)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(classes)
        
        ax.set_ylim(top=-0.5, bottom=-0.5+len(classes))
        ax.set_xlim(left=-0.5, right=-0.5+len(classes))
        
        fmt = '.2f'
        threshold = (np.nanmax(matrix)+np.nanmin(matrix)) / 1.5
        r=matrix.shape[0]
        c=matrix.shape[1]
    
        for i, j in itertools.product(range(r), range(c)):
            ax.text(j, i, format(matrix[i, j], fmt)+"%",
                     horizontalalignment="center",
                     color="white" if (matrix[i, j] > threshold) else "black")
    
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        
        
        return fig.axes
    
    @_recordFunctionCall
    def plotMatrix(self, matrix, classes, title='Confusion matrix', 
                            cmap=None, figsize=None, ylabel='True classes', xlabel='Predicted classes',
                            save=False, buffer_stream=None, filepath="confusion_matrix.png", timestampFilename=True):
        
        fileFormat=self._getFileType(filepath)
        if fileFormat==None: raise Exception("No file format was detected. Check the filepath.")
        
        axes=self._drawMatrix(matrix, classes, title=title,
                                   cmap=cmap, figsize=figsize,ylabel=ylabel, xlabel=xlabel)      
        ax=axes[0]
        fig=ax.get_figure()
        
        #add a colorbar.
#        axImg=ax.get_images()[0]
#        fig.colorbar(axImg)
        
        fig.tight_layout()
        
        if timestampFilename is True:
            filepath=self._addTimestampToFilename(filepath)
        
        if save is True: fig.savefig(filepath)
        
        if buffer_stream != None:
            fig.savefig(buffer_stream, format='png')
            pyplot.close(fig)
            return None
        
        fig.show()
        
        return None
    
    @_recordFunctionCall
    def plotVectorAndMatrix(self, vector, matrix, classes, 
                                        title="Precision and Confusion Matrix", 
                                        cmap=None, figsize=[15,10], save=False, 
                            filepath="precision_and_confusionMatrix.png", timestampFilename=True):
        
        cmAxes=self._drawMatrix(matrix=matrix, classes=classes,
                                   cmap=cmap, figsize=[7,5])

        pAxes=self._drawVector(vector=vector, classes=classes,
                                   cmap=cmap, figsize=[7,5], ylabel="")
        
#        temp=TransformNode()
    
        newFig = copy.deepcopy(cmAxes[0].get_figure())
        for ax in newFig.axes:
                newFig.delaxes(ax)
        
        allAxes=cmAxes+pAxes
        
        gs = gridspec.GridSpec(1, 4, width_ratios=[20,1,5,1], height_ratios=[1], wspace=0.35)
        newAxes=[]
        for i,_ in enumerate(gs):
            ax = newFig.add_subplot(gs[i])
            newAxes.append(ax)
            
#        newFig.set_figwidth(figsize[0])
#        newFig.set_figheight(figsize[1])
        
        for i, ax in enumerate(allAxes):
            oldFig=ax.get_figure()
            ax.figure=newFig
            oldFig.delaxes(ax)
            newFig.axes.append(ax)
            newFig.add_axes(ax)
            ax.set_position(newAxes[i].get_position())
            newAxes[i].remove() # remove the replaced axes.


        pAxes[0].get_shared_y_axes().join( pAxes[0], cmAxes[0])
        pAxes[0].tick_params(axis='y',which='both', left=False, right=False,
             labelleft=False, labelright=False) 
#        newFig.colorbar(cmAxImg, ax=cmAx)
#        newFig.colorbar(pAxImg, ax=pAx)                                

        newFig.suptitle(title)
        
        if timestampFilename is True:
            filepath=self._addTimestampToFilename(filepath)

        if save is True: newFig.savefig(filepath)
        
        
        newFig.show()
        
        return None
    
    def generateBoxplotOfScores(self, listOfScores, correspLabels, ylabel=None, 
                                showmeans=True, cmapName="hsv", 
                                save=False, filepath="scores.png", timestampFilename=True):
        fig=pyplot.figure()
        ax=fig.add_subplot(1,1,1)
        
        myBoxplot=ax.boxplot(listOfScores, labels=correspLabels, patch_artist=True, showmeans=showmeans)
        myColormap=cm.get_cmap(name=cmapName)
        for i,box in enumerate(myBoxplot["boxes"]):
            box.set(facecolor=myColormap(i/len(myBoxplot["boxes"])))
        
        # remove any x-axix tick marks, labels, etc.
        ax.tick_params(axis='x',which='both', bottom=False, top=False, labelbottom=False)
        
        ax.set_ylabel(ylabel)
        ax.set_ylim([0,1])
        ax.legend(myBoxplot["boxes"],correspLabels,bbox_to_anchor=(1,1))
        fig.tight_layout()
        
        if timestampFilename is True:
            filepath=self._addTimestampToFilename(filepath)

        if save is True: fig.savefig(filepath)
        
        fig.show()
        
        return ax
    
    @_recordFunctionCall
    def plotTimeSeries(self, df, colNameOfTime, variables, locsOfSpans=None, 
                       cmapForSpans="nipy_spectral",cmapForTimeSeries="nipy_spectral", fig_title=None,
                       marker=None, figsize=None, orientation="v",invert_xaxis=False, invert_yaxis=False,save=False,
                        filepath="time_series.png", timestampFilename=False):
        '''
        Operation:
            Plots time series, and optionally plots spans across them. 
            
            Note that if plotting spans as well,
            then the start and end of each span must occur at different
            timestamps for the span to show up.
            
        Paramters:
            
            **df**: pandas.DataFrame of the dataset.
            
            **colNameOfTime**: String. The label of the time column.
            
            **variables**: List of strings, or list of lists of strings, that 
            represent the variables to be plotted. For example, ["var1", ["var2","var3","var4"]]
            will result in two subplots on the figure, where var1 is plotted on the first subplot, while 
            var2, var3 and var4 are all plotted together on the second subplot.
            
            **locsOfSpans**: Dict of lists. The start and the end indices for all contiguous spans.
                        
            **cmapForSpans**: String. Name of matplotlib colormap to use for the horizontal spans.
            
            **cmapForTimeSeries**:  String. Name of matplotlib colormap to use for the time series curves.

            **marker**:            
            
            **figsize**: tuple of the form (width, height) that represents figure dimension.
            
            **orientation**: String. Options are "v" for vertical or "h" for horizontal.
            
            **invert_yaxis**: True or False.
            
            **invert_xaxis**: True or False.
            
            **save**: Boolean. Set to True to save figure to current directory. 
            To save to a different directory, supply the path to the filepath argument.
            
            **filepath**: String. Directory (including filename) to save to. 
            Default is "time_series.png".
            
            **timestampFilename**: Boolean. Set to True to include the timestamp of 
            when the figure was saved.
        '''
        if fig_title==None: fig_title=""
        # unpack the info from the args, and repack for plotting.
              
        cmapForTimeSeries=cm.get_cmap(name=cmapForTimeSeries)
        norm = colors.Normalize(vmin=0, vmax=len(variables))
        
        plotInfo=copy.deepcopy(variables)
        for i,element in enumerate(plotInfo):
            if isinstance(element,list):
                for ii,elem in enumerate(element):
                    element[ii]=[elem,colNameOfTime, cmapForTimeSeries(norm(ii))]
            else:
                plotInfo[i]=[element,colNameOfTime, cmapForTimeSeries(norm(i))]
        
        # set up the plot and plot the data.
        
        subplot_shape=[1,len(plotInfo)]
        if orientation=="h":
            subplot_shape=subplot_shape[::-1] # reverse the order.
        
        if figsize==None: 
            figsize=(len(plotInfo)*2,20)     
        if orientation=="h":
            figsize=figsize[::-1] # reverse the order.
                
        fig, subplots = pyplot.subplots(nrows=subplot_shape[0], ncols=subplot_shape[1], figsize=figsize)
        
        if not self._isIterable(subplots):
            # If not an iterable, i.e. only one subplot (Axes).
            # Put in a list.
            subplots=[subplots]        
        
        for (i,curveInfo) in enumerate(plotInfo):
            
            ax=subplots[i]
            for elem in curveInfo:
                if isinstance(elem, list):
                    if orientation=="v":
                        ax.plot(df.loc[:,elem[0]],df.loc[:,elem[1]],color=elem[2],marker=marker)
                    if orientation=="h":
                        ax.plot(df.loc[:,elem[1]],df.loc[:,elem[0]],color=elem[2],marker=marker)
                    
                    ax.set_title(elem[0])
                    #TODO: show titles of all curves in the same plot (if aestheically satisfactory).
                else:
                    if orientation=="v":
                        ax.plot(df.loc[:,curveInfo[0]],df.loc[:,curveInfo[1]],color=curveInfo[2],marker=marker)
                    if orientation=="h":
                        ax.plot(df.loc[:,curveInfo[1]],df.loc[:,curveInfo[0]],color=curveInfo[2],marker=marker)
                    
                    ax.set_title(curveInfo[0])
            if i>0: # hide axis labels for all except the first subplot.
                if orientation=="v":
                    ax.tick_params(axis='y',which='both', right =False, left=False, labelleft=False)
                if orientation=="h":
                    ax.tick_params(axis='x',which='both', top =False, bottom=False, labelbottom=False)
            if invert_yaxis==True:
                ax.invert_yaxis()
            if invert_xaxis==True:
                ax.invert_xaxis()
        # plotting the spans.
        
        colorPatches=[]
        
        if locsOfSpans!=None:
                # setting up colors.
                cmap=cm.get_cmap(name=cmapForSpans)
                norm = colors.Normalize(vmin=0, vmax=len(locsOfSpans.keys())) 
                # plot the spans.
                for i, class_ in enumerate(locsOfSpans.keys()):
                    clsColor=cmap(norm(i)) # Color is a tuple in the form of (R,G,B,A).
                    ## Change alpha. 
                    clsColor=list(clsColor)
                    clsColor[3]=0.3 # new alpha value.
                    clsColor=tuple(clsColor)
                    for ax in subplots:
                        for start, end in locsOfSpans[class_]:
                            if orientation=="v":
                                ax.axhspan(df.loc[start,colNameOfTime], df.loc[end,colNameOfTime], 
                                       color=clsColor)
                            if orientation=="h":
                                ax.axvspan(df.loc[start,colNameOfTime], df.loc[end,colNameOfTime], 
                                       color=clsColor)                                
                    # create legend for the spans.
                    colorPatch = patches.Patch(color=clsColor, label=class_)
                    colorPatches.append(colorPatch)                
        
        fig.legend(handles=colorPatches)
        fig.suptitle(fig_title)
    	# saving figure.
        
        if timestampFilename is True:
            filepath=self._addTimestampToFilename(filepath)
        
        if save is True: fig.savefig(filepath)
        
        fig.show()
        return None
    
    @_recordFunctionCall
    def plotAllClassClusters2D(self, df, classColName, classesAndThresholds, xColName, yColName, 
                               xAxisRange, yAxisRange, xlabel=None, ylabel=None, markerSize=1,  
                               colorScheme=cm.Reds, save=False, filepath="class_clusters.png", 
                               timestampFilename=False):
        
        if xlabel is None: xlabel=xColName            
        if ylabel is None: ylabel=yColName
        
        fig=pyplot.figure()
        ax=fig.add_subplot(1,1,1)
        colors=colorScheme(np.linspace(0,1,len(classesAndThresholds)))
    
        classes=[]
        i=0
        for (cls, startThresh) in classesAndThresholds:
            x=df[df[classColName]==cls][xColName]
            y=list(df[df[classColName]==cls][yColName].get_values())
            ax.scatter(x, y, c=colors[i], s=markerSize)
            classes.append(cls)
            i+=1
        
        ax.legend(classes)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(xAxisRange)
        ax.set_ylim(yAxisRange)
        
        if timestampFilename is True:
           filepath=self._addTimestampToFilename(filepath)
    
        if save is True: fig.savefig(filepath)
    
        fig.show()
        
        return ax
    
    @_recordFunctionCall
    def plotEachClassCluster2D(self, df, classColName, classesAndThresholds, xColName, yColName, 
                               xAxisRange, yAxisRange, markerSize=1, figsize=None, colorScheme=cm.Reds,
                               save=False, filepath="class_cluster.png", timestampFilename=False):
        
        numOfSubplots=len(classesAndThresholds)
        
        if figsize is None: figsize=[5, numOfSubplots*5] # note: [w, h].
    
        colors=colorScheme(np.linspace(0,1,len(classesAndThresholds)))
        
        fig, axes = pyplot.subplots(numOfSubplots, 1, figsize=figsize)      
        axes=axes.flatten()
        i=0
        for (cls, startThresh) in classesAndThresholds:
            axes[i]=df[df[classColName]==cls].plot(kind="scatter", x=xColName, y=yColName, 
                                           c=colors[i], s=markerSize, title=cls,
                                           ylim=yAxisRange, xlim=xAxisRange, ax=axes[i])
            i+=1
            
        filepath=filepath.replace(".png","")+cls+".png"
        
        if timestampFilename is True:
            filepath=self._addTimestampToFilename(filepath)
            
        if save is True: fig.savefig(filepath)
        fig.show(warn=False)            
                
        return axes
#%%
        
#d=pd.DataFrame({"col1":(["a"]*15+["b"]*11),"col2":np.random.rand(26), "col3":np.random.rand(26)})
#c=Tools().oneHotEncode(df=d,colName="col")
#
#clss=np.array(["a","b","c"])
#p=np.array([.5,.3,.2])
#Tools().plotPrecision(p, classes=clss)
#from sklearn import naive_bayes
#myModel=naive_bayes.GaussianNB(priors=None)
#
#acc,ap=Tools().kFoldCrossValidate(d, ["col2"], "col1", model=myModel,n_splits=10, scorings=["accuracy","average_precision"])

#g=Tools().getPortion(df=d,proportion=[0.3,0.6],start_from="middle")

#%%
#fig=pyplot.figure()
#ax=fig.add_subplot(1,1,1)
#ax.set_title("AAA")
#scatter=ax.scatter(d["col2"],d["col3"])
##ax=d.plot(kind="scatter", x="col2", y="col3", ax=ax)

#%%
#a.get_array()
