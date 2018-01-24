# APS490Y1_2017
Python Code
#!/usr/bin/python
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.types import *
from pyspark.sql.types import IntegerType
import time
import sys

def GenerateFieldTableJson(numFields=0, numRows=0, initVal=0):
	"""Generates a JSON file containing the definition of an SQL table in json format.
		File is generated in the directory where the script is being run from.
		All fields initialized to initVal.
		Default inputs (integers) are (0,0,0) for inputs (numFields,numRows,initVal). 
	"""
	writeStr = "" #The string to be written to the JSON file
	with open("generatedFieldTable.json","w") as json:
		for i in range(numRows):
			writeStr += "{"
			for j in range(numFields):
				if (j != numFields-1):
					writeStr += '"Field{0}":{1},'.format(j,int(initVal)) #All fields initialized to initVal
				else:
					writeStr += '"Field{0}":{1}'.format(j,int(initVal)) #Remove last comma	
			writeStr += "}\n"
		json.write(writeStr)

def mod(x,y):
	return x % y

def mbt1(spark, fieldSel=0):
	#Convert the JSON file into a DataFrame pointer
	df = spark.read.json("generatedFieldTable.json")
	df.printSchema()
  
  #Run an integer operation in place
	df.select((df["Field"+str(fieldSel)]%(sys.maxint-2)).alias("In Place Arithmetic Operation")).show() #Field % 8 benchmark
	
	#Run an arbitrary function on the DataFrame contents
	df.select(mod(df["Field"+str(fieldSel)],(sys.maxint-2)).alias("Arbitrary Function Result")).show()
	
if __name__ == "__main__":
  #Create a new Spark Session or get 
  #a pointer to one if it already exists
	spark = SparkSession \
		.builder \
		.appName("Python Spark Task 1 Microbenchmarking") \
		.config("spark.some.config.option", "some-value") \
		.getOrCreate()
  #Generate a file to be loaded as a DataFrame
  #object into the Spark Session
	GenerateFieldTableJson(128,128,0)
  #Run the mbt1 function in the Spark Session
	mbt1(spark)	
  #Keep the localhost connection from being
  #closed due to the spark session closing.
	time.sleep(1000)
  #Close the spark session.
	spark.stop()
