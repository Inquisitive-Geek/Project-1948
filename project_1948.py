import re
# import string
from docx import Document
import csv
import mmap
import re
#Prepare function to read in documents
def read_doc_file_to_string(file_name, read_mode):
    f = open(file_name, read_mode)
    document = Document(f)
    f.close()
    fulltext = []
    for para in document.paragraphs:
        fulltext.append(para.text)
    fulltext_str = ' '.join(fulltext)
    return fulltext_str

#removes any text contained inside parentheses or brackets.
def remove_grouped_text(input_text):
    open_index = input_text.find('(')
    close_index = input_text.find(')')
    while open_index > 0:
        input_text = input_text[:open_index]+input_text[(close_index+1):]
        open_index = input_text.find('(')
        close_index = input_text.find(')')
    open_index = input_text.find('[')
    close_index = input_text.find(']')
    while open_index > 0:
        input_text = input_text[:open_index]+input_text[(close_index+1):]
        open_index = input_text.find('[')
        close_index = input_text.find(']')
    return input_text

#remove punctation characters
def remove_punctuation_characters(input_text):
    processed_text = input_text.replace("P:"," INTERVIEWEE ")
    processed_text = processed_text.replace("I:", " INTERVIEWER ")
    processed_text = re.sub(r"[^a-zA-Z0-9]"," ",processed_text)
    return processed_text

#function to make case of words uniform and puts one space between each word.
    #input_text is the interview as a string
    #upper (optional) is True if the words are to be upper case
    #lower (optional) is True if the words are to be lower case
    #title (optional) is True if the words are to have 'title' case
def normalize_text(input_text,upper=False,lower=False,title=False,sep=" "):
    words = input_text.split()
    output_text=""
    if upper:
        for word in words:
            word.upper()
    if lower:
        for word in words:
            word.lower()
    if title:
        for word in words:
            word.title()
    output_text=sep.join(words)
    return output_text
        
#function to separate out the statements of the interview.
    #input_text is the interview as a string
    #interviewer is the string in the text that marks when interviewer is speaking
    #interviewee is the string in the text that marks when interviewee is speaking.
def separate_statements(input_text, interviewer, interviewee):
    j = 0
    interviewer_text = ""
    interviewee_text = ""
    text_partition = []
    while (interviewer in input_text) or (interviewee in input_text):
        if j%2==0:
            text_partition = input_text.partition(interviewee)
            input_text = text_partition[2]
            interviewer_text = interviewer_text+'NEXT'+text_partition[0]
        if j%2==1:
            text_partition = input_text.partition(interviewer)
            input_text = text_partition[2]
            interviewee_text = interviewee_text+'NEXT'+text_partition[0]
        j=j+1
    if j%2==0:
       interviewer_text = interviewer_text+'NEXT'+input_text
    if j%2==1:
        interviewee_text = interviewee_text+'NEXT'+input_text
    return [interviewer_text,interviewee_text]

#function to find a word and the phrase it is a part of.
    #input_text is text to be analyzed
    #word_list is a list of words to be looked for
    #radius is the number of words to be taken before and after search word for context
def word_context(input_text, word_list, radius):
    words = input_text.split()
    output = []
    if type(word_list)==type(output):
        for word in word_list:
            word_neighborhoods = []
            for compare in words:
                if word.title() == compare.title():
                    center = words.index(compare)
                    neighborhood = words[center-radius:center+radius+1]
                    word_neighborhoods.append(string.join(neighborhood))
            output.append(word_neighborhoods)
    elif type(word_list)==type(""):
        word_neighborhoods = []
        for compare in words:
            if word_list.title() == compare.title():
                center = words.index(compare)
                neighborhood = words[center-radius:center+radius+1]
                word_neighborhoods.append(string.join(neighborhood))
        output.append(word_neighborhoods)
    else:
        print("ERROR IN TYPE")
    return output

#function to clean the data
    #text_arr is the input array of strings -- each string is a transcript of the interview
    #cleaned_text_arr is the outputs array of cleaned strings
def clean_text(text_arr):
    cleaned_text_arr = []
    for text in text_arr:
        # Remove grouped text like (laughing) etc.
        text = remove_grouped_text(text)
        # Remove punctuation characters and replace P: & I: with INTERVIEWER & INTERVIEWEE
        text = remove_punctuation_characters(text)
        # Normalize the input text
        text = normalize_text(text)
        cleaned_text_arr.append(text)
    return cleaned_text_arr


##Next we will count keywords and relevant phrases.

def count_keywords_and_phrases(list_of_interviews):
    keywords_and_phrases = []
    with open('Keywords_And_Phrases.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            keywords_and_phrases.append(row[0])
    #keywords are already stored in a csv
    percentage_used = [] 
    for word in keywords_and_phrases:
        sum_used = 0
        for interview in list_of_interviews:
            if ((word in interview) or (word.lower() in interview)):
                sum_used += 1
        percentage = (sum_used/len(list_of_interviews)) * 100
        #this will tell us how many interviews this particular word was used in
        percentage_used.append(percentage)
        #each word will have a corresponding percentage so word_0 has a
        #percentage of percentage_0
    quicksort(percentage_used, keywords_and_phrases, 0,
              len(percentage_used) - 1)
    #sort so it will be easier to see what words are used most frequently
    #as the percentages are sorted so is the array of words so each words
    #percentage is still easy to find
    word_and_percentage_used = open("word_and_percentage_used.csv", "w")
    word_and_percentage_used.write("Keyword,Percentage\n")
    
    for i in range(len(percentage_used) - 1, 0, -1):
        word_and_percentage_used.write(keywords_and_phrases[i] + ',' +
                                    str(percentage_used[i]) + '\n')
    word_and_percentage_used.close()
    print("finished")

def count_words(list_of_interviews):
    keywords_and_phrases = []
    with open('Keywords_And_Phrases.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            keywords_and_phrases.append(row[0])
    frequency = []
    for i in range(len(keywords_and_phrases)):
        frequency.append(0)
    for interview in list_of_interviews:
        for i in range(len(keywords_and_phrases)):
            frequency[i] += interview.count(keywords_and_phrases[i])
            frequency[i] += interview.count(keywords_and_phrases[i].lower())
    
    quicksort(frequency, keywords_and_phrases, 0, len(frequency) - 1)
    word_and_freq = open("word_and_frequency.csv", "w")
    word_and_freq.write("keyword,frequency\n")
    for i in range(len(frequency) - 1, 0, -1):
        word_and_freq.write(keywords_and_phrases[i]
                            + ',' + str(frequency[i]) + '\n')
    word_and_freq.close()
    print("finished")
    
#Standard quicksort algorithm just had to tweak so it will sort two arrays
#at a time
def quicksort(list, list_2, min, max):
    if (min < max):
        p = partition(list, list_2, min, max)
        quicksort(list, list_2, min, p - 1)
        quicksort(list, list_2, p + 1, max)

def partition(list, list_2, min, max):
    pivot = list[max]
    i = min - 1
    for j in range(min, max):
        if list[j] < pivot:
            i += 1
            temp =list[j]
            temp2 = list_2[j]
            list[j] = list[i]
            list_2[j] = list_2[i]
            list[i] = temp
            list_2[i] = temp2
    if list[max] < list[i + 1]:
        temp = list[max]
        temp2 = list_2[max]
        list[max] = list[i + 1]
        list_2[max] = list_2[i + 1]
        list[i + 1] = temp
        list_2[i + 1] = temp2
    return i + 1
        
    
        
            
                


            
