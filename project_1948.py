import re
import string
import numpy as np
import pandas as pd
from docx import Document

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
def normalize_text(input_text,upper=False,lower=False,title=False):
    words = input_text.split()
    output_text=""
    if upper:
        j=0
        for word in words:
            words[j] = word.upper()
            j=j+1
    if lower:
        j=0
        for word in words:
            words[j] = word.lower()
            j=j+1
    if title:
        j=0
        for word in words:
            words[j] = word.title()
            j=j+1
    output_text=string.join(words)
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

#function to break up test string by sentences
    #input_text is text to be split
def sentences(input_text):
    new_text = string.replace(input_text,'?','.')
    new_text = string.replace(new_text,'!','.')
    sentence_list = []
    index_end = new_text.find('. ')
    while index_end>=0:
        sentence_list.append(new_text[:index_end+1])
        new_text = new_text[index_end+1:]
        index_end = new_text.find('. ')
    if new_text not in string.whitespace:
        sentence_list.append(new_text)
    return sentence_list
    
#function to count how many times pairs of words occur in the same sentence
#entry i,j of returned data frame is number of sentences containing word i and word j 
#entry i,i of returned data frame is number of sentences contaiing word i
    #sentence_list is a list containing the sentences of the text
    #words is a list containing the specific words being correlated.
def word_pairs(sentence_list, words):
    np_zeros = np.zeros(shape=(len(words),len(words)))
    word_pairs = pd.DataFrame(data=np_zeros, columns = words, index = words)
    for outer_word in words:
        for sentence in sentence_list:
            if outer_word in sentence:
                for inner_word in words:
                    if inner_word in sentence:
                        old_count = word_pairs.get_value(outer_word,inner_word)
                        word_pairs.set_value(outer_word,inner_word,old_count+1)
    return word_pairs

#function to determine how many sentences contain a set list of words
    #sentence_list is a list containing sentences of text
    #words is a list of the words being searched for
def words_in_sentences(sentence_list, words):
    count_main = 0
    for sentence in sentence_list:
        j=0
        for word in words:
            if word in sentence:
                j=j+1
        if j==len(words):
            count_main = count_main+1
    return count_main
