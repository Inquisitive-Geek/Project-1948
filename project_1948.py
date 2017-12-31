import re
import string
import numpy as np
import pandas as pd
from docx import Document
import csv

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
def remove_punctuation_characters(input_text, interviewer = " INTERVIEWER ",interviewee = " INTERVIEWEE "):
    processed_text = input_text.replace("P:",interviewee)
    processed_text = processed_text.replace("I:",interviewer)
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
def separate_statements(input_text, interviewer, interviewee, breaks = True, break_mark = "NEXT"):
    j = 0
    interviewer_text = ""
    interviewee_text = ""
    text_partition = []
    while (interviewer in input_text) or (interviewee in input_text):
        if j%2==0:
            text_partition = input_text.partition(interviewee)
            input_text = text_partition[2]
            if breaks:
                interviewer_text = interviewer_text + break_mark + text_partition[0]
            else:
                interviewer_text = interviewer_text + text_partition[0]
        if j%2==1:
            text_partition = input_text.partition(interviewer)
            input_text = text_partition[2]
            if breaks:
                interviewee_text = interviewee_text + break_mark + text_partition[0]
            else:
                interviewee_text = interviewee_text + text_partition[0]
        j=j+1
    if j%2==0:
        if breaks:
            interviewer_text = interviewer_text + break_mark + input_text
        else:
            interviewer_text = interviewer_text + input_text
    if j%2==1:
        if breaks:
            interviewee_text = interviewee_text + break_mark + input_text
        else:
            interviewee_text = interviewee_text + input_text               
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
    new_text = string.replace(input_text, '..',' ')
    new_text = string.replace(input_text, '...', ' ')
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
#entry i,i of returned data frame is number of sentences containing word i
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

#function to find sentences containing all or subset of a list of words
    #sentence_list is a list containing sentences of text
    #words is a list of the words being searched for
def sentences_with_words(sentence_list, words, contain_all = True, contain_min = 1):
    sentences = []
    if contain_all:
        for sentence in sentence_list:
            j=0
            for word in words:
                if word in sentence:
                    j=j+1
                if word not in sentence:
                    break
            if j==len(words):
                sentences.append(sentence)
    else:
        for sentence in sentence_list:
            j=0
            for word in words:
                if word in sentence:
                    j=j+1
            if j>=contain_min:
                sentences.append(sentence)
    return sentences

#--MOST RECENT PULLS FROM GITHUB--
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

# With the use of regular expressions, the text belonging to only Interviewees is extracted and combined 
# 2 outputs are returned:
# one output is just the list of interviewee responses - txt_all
# other output is list converted to text blob with commas removed and all words converted to lowercase - txt_combined
def extract_interviewee(text_all, extracting_string, terminating_string):
    import re
    txt_all = []
    for txt in text_all:
        s =re.split(extracting_string, txt)
        for i in range(0,(len(s))):
            t = re.split(terminating_string, s[i])[0]
            txt_all.append(t)
    #The text belonging to all Interviewee's are joined together            
    txt_combined = ','.join(txt_all)
    # All the comma strings are replaced with no space
    txt_combined = txt_combined.replace(',', '')
    # All the text is converted to its lower case
    txt_combined = txt_combined.lower()  
    return txt_all, txt_combined


# Using PyDictionary creating the synonym table 
# Creating the list of synonyms
def synonym_collect(text,keyword_list):
    from PyDictionary import PyDictionary
    dictionary = PyDictionary()
    synonym_table = {}
    for keyword in keyword_list:
        if len(keyword.split(' ')) == 1:
            if (dictionary.synonym(keyword) != None):
                synonym_list = dictionary.synonym(keyword)
                synonym_table[keyword] = {len(re.findall(keyword.lower(), text)):
                                      {synonym.lower():len(re.findall(synonym.lower(), text)) for synonym in synonym_list}}
    return synonym_table
        
# Creating wordcloud for the words in the interviewees responses after removing stopwords
def create_wordcloud(text):
    from wordcloud import WordCloud, STOPWORDS
    stopwords = set(STOPWORDS)
    additional_stopwords = ['one','see','yes','really','yeah','maybe','say','know','think','well','lot','make','will','also','don','going',
                           'go','something','everything']
    for new_word in additional_stopwords:
        stopwords.add(new_word)
    wc = wordcloud.WordCloud(stopwords = stopwords).generate(text)
    return wc              

#function to determine proportional frequency of a list of phrases.
def proportional_phrase_use(text,phrases):
    text = text.lower()
    phrases = [phrase.lower() for phrase in phrases]
    phrases.sort(key = lambda phrase: len(phrase),reverse=True)#sort phrases by decreasing length
    phrase_usage = {}
    usage_count = 0
    for phrase in phrases:
        phrase_usage[phrase]=len(re.findall(phrase,text))
        for longer_phrase in phrases[:phrases.index(phrase)]:
            if phrase in longer_phrase:
                phrase_usage[phrase] = phrase_usage[phrase]-phrase_usage[longer_phrase]
        usage_count = usage_count+phrase_usage[phrase]
    total_words = float(len(text.split()))
    return usage_count/total_words

def phrase_spread(text, phrases, num_sections):
    print('Change 2')
    partition = np.linspace(0,len(text),num_sections+1,endpoint=True)
    partition = [int(np.round(p)) for p in partition]
    print(partition)
    text = text.lower()
    phrases= [phrase.lower() for phrase in phrases]
    phrases.sort(key=lambda phrase: len(phrase),reverse=True)
    sections = {}
    phrase_appearance = {}
    for j in range(0,num_sections):
        appearance_count = 0
        text_section=text[partition[j]:partition[j+1]+1]
        for phrase in phrases:
            print('working with'+phrase)
            phrase_appearance[phrase]=len(re.findall(phrase,text_section))
            for longer_phrase in phrases[:phrases.index(phrase)]:
                if phrase in longer_phrase:
                    phrase_appearance[phrase] = phrase_appearance[phrase]-phrase_appearance[longer_phrase]
            appearance_count = appearance_count+phrase_appearance[phrase]
            print(appearance_count)
        sections[j]=appearance_count
    return sections