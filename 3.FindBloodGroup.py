import cv2
from matplotlib import pyplot as plt
import sys
import os
import imutils
#from scipy.misc import imread
from scipy.linalg import norm
from scipy import sum, average
from matplotlib.pyplot import imread



def to_grayscale(arr):
    "If arr is a color image (3D array), convert it to grayscale (2D array)."
    if len(arr.shape) == 3:
        return average(arr, -1)  # average over the last axis (color channels)
    else:
        return arr

def normalize(arr):
    rng = arr.max()-arr.min()
    amin = arr.min()
    return (arr-amin)*255/rng


print('Enter filename :')
fname=input()
#img1 = cv2.imread('./' + fname,0)

i=1
dim=[100,100]
#img1 =imutils.resize(img1, width=100,height=100) #cv2.resize(img, dim)#, interpolation = cv2.INTER_AREA)
#img1 = cv2.resize(img1, dim, interpolation = cv2.INTER_AREA)
#Creating training data - pixel values of yes images to csv file
cnt=0
st1=''
img=cv2.imread('./' + fname,0) #(imread('./' + fname).astype(float))
#img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
img =cv2.resize(img, (100,100))#img =imutils.resize(img, width=100,height=100) #cv2.resize(img, dim)#, interpolation = cv2.INTER_AREA)
siz= img.shape[0] * img.shape[1]
st1='1,'
for i in range(0,img.shape[0]):
   for j in range(0,img.shape[1]):
         #print( int(img[i,j]),sep=' ')
         st1=st1 + str(int(img[i,j])) +','
st1=st1[:-1]
sttmp=st1
#print(st1)

#Check in csv
file1 = open("fingerprint_train.csv","r")
lines = file1.readlines()
#print(len(lines))

#if(lines[1][0:20000]==st1[0:20000]):
#   print('yes')

found=0
cnt=0
for i in range(1,len(lines)):
   cnt=cnt+1
   lines[i]=lines[i].strip("\n")
   wrds= lines[i].split(",")
   #print(wrds)
   wrd1=wrds[len(wrds)-1]
   wrd=wrds[len(wrds)-2]
   #print(wrd)
   
   l=lines[i] #[0:lines[i].rindex(',')]
   #print(l)
   st1=sttmp + "," + wrd + "," + wrd1
   st1=st1.rstrip(",")
   #print(st1)

   #if len(l)== len(st1):

   count1= len(l.split(","))
   tok1=l.split(",")
   count2=len(st1.split(","))
   tok2=st1.split(",")
   cntmatch=True
   if count1-count2==1:
         st1 =st1 + "," + ""
   if count2-count1==1:
         l=l+ "," + wrd
   count1= len(l.split(","))
   tok1=l.split(",")
   count2=len(st1.split(","))
   tok2=st1.split(",")
   #print(count1)
   #print(count2)
   tmpcount=0
   #print(count1-2)
   for j in range(0,count1-2):
      if (tok1[j] == tok2[j]):
         tmpcount=tmpcount+1
   #print(tmpcount)

   
   for j in range(0,count1-2):
      if (tok1[j] != tok2[j]):
         cntmatch=False
         #print(tok1[j])
         #print(tok2[j])
   if cntmatch==True:
       print('Matched in ' + str(cnt)+ ' Positive Blood Group found.')
       print(tok1[len(tok1)-2])
       found=1
       #break
   #print(count1) 
   #print(count2) 
   #exit()

   #print(len(l))
   #print(len(st1))
   if(len(l)-1==len(st1)):
     if(l[0:-1]==st1):
     #if(l==st1):
       print('Matched in Record ' + str(cnt-1)+ ' Positive Blood Group found.')
       print(tok1[len(tok1)-2])
       found=1
       break



st1='0,'
for i in range(0,img.shape[0]):
   for j in range(0,img.shape[1]):
         #print( int(img[i,j]),sep=' ')
         st1=st1 + str(int(img[i,j])) +','
st1=st1[:-1]
sttmp=st1

if (found==0):
 cnt=0
 for i in range(1,len(lines)):
   cnt=cnt+1
   lines[i]=lines[i].strip("\n")
   wrds= lines[i].split(",")
   #print(len( wrds[len(wrds)-1]))
   #exit()
   wrd1=wrds[len(wrds)-1]
   wrd=wrds[len(wrds)-2]
   #print(wrd)
   
   l=lines[i] #[0:lines[i].rindex(',')]
   #print(l)
   st1=sttmp + "," + wrd + "," + wrd1
   st1=st1.rstrip(",")
   #print(st1)

   #if len(l)== len(st1):

   count1= len(l.split(","))
   tok1=l.split(",")
   count2=len(st1.split(","))
   tok2=st1.split(",")
   cntmatch=True
   if count1-count2==1:
         st1 =sttmp + "," + wrd
   if count2-count1==1:
         l=l+ "," + wrd

   count1= len(l.split(","))
   tok1=l.split(",")
   count2=len(st1.split(","))
   tok2=st1.split(",")
   #print(count1)
   #print(count2)

   for j in range(0,count1-2):
      if (tok1[j] != tok2[j]):
         cntmatch=False
         #print(tok1[j])
         #print(tok2[j])
   if cntmatch==True:
       print('Matched in ' + str(cnt)+ ' Negative Blood Group found.')
       print(tok1[len(tok1)-2])
       found=1
       break
       print(count1) 
       print(count2) 
       print(len(l))
       print(len(st1))
       tok1=l.split(",")
       #print(tok1)
       #tok2=st1.split(",")
       #print(tok2)
   if(len(l)-1==len(st1)):
     if(l[0:-1]==st1):
     #if(l==st1):
       print('Matched in Record ' + str(cnt-1)+ ' Negative Blood Group found.')
       print(tok1[len(tok1)-2])
       found=1
       break
   if(len(l)-2==len(st1)):
     if(l[0:-1]==st1):
     #if(l==st1):
       print('Matched in Record ' + str(cnt-1)+ ' Negative Blood Group found.')
       found=1
       break
"""
 found=0
 cnt=0
 st1='0,'
 for i in range(0,img.shape[0]):
   for j in range(0,img.shape[1]):
         #print( int(img[i,j]),sep=' ')
         st1=st1 + str(int(img[i,j])) +','
 st1=st1[:-1]
 for i in range(1,len(lines)):
   cnt=cnt+1
   wrds= lines[i].split(",")
   #print(wrds)
   wrd1=wrds[len(wrds)-1]
   wrd=wrds[len(wrds)-2]
   #print(wrd)
   
   l=lines[i] 
   l=l.strip("\n")
   #print(l)
   st1=st1 + "," + wrd + wrd1
   print(st1)
   exit()
   wrds= l.split(",")
   if(len(l)-1==len(st1)):
     #print(len(l)-1)
     if(l[0:-1]==st1):
       print('Matched in Record ' + str(cnt-1) + ' Negative Blood Group found.')
       found=1
       break
"""
if(found==0):
 print('Not Matched in any of the blood groups')

