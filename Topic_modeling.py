import project_1948 as pjct
import nltk
from gensim import corpora, models
import docx as w
nltk.download('stopwords')

#pull interview files
interview_1 = pjct.read_doc_file_to_string('Interview #1 - Participant 1.docx','rb')
interview_2 = pjct.read_doc_file_to_string('Interview #2-Participant 2.docx','rb')
#interview_4 = pjct.read_doc_file_to_string(folder+'Interview#4Participant#4.docx','rb')
interview_5 = pjct.read_doc_file_to_string('Interview #5. Participant # 5.docx','rb')
interview_6 = pjct.read_doc_file_to_string('Interview #6. Participant #6.docx','rb')
interview_7 = pjct.read_doc_file_to_string('Interview #7. Participant #7..docx','rb')
interview_9 = pjct.read_doc_file_to_string('Interview #9. Participant #9.docx','rb')
interview_10 = pjct.read_doc_file_to_string('Interview #10. Participant #10.docx','rb')
interview_11 = pjct.read_doc_file_to_string('Interview #11 (1).docx','rb')
interview_12 = pjct.read_doc_file_to_string('Interview #12.docx','rb')
interview_13 = pjct.read_doc_file_to_string('Interview #13. Participant #13.docx','rb')

interview_14 = pjct.read_doc_file_to_string('interviews_new/Adnan.docx', 'rb')
interview_15 = pjct.read_doc_file_to_string('interviews_new/Amer.docx', 'rb')
interview_16 = pjct.read_doc_file_to_string('interviews_new/Aris.docx', 'rb')
interview_17 = pjct.read_doc_file_to_string('interviews_new/Belma.docx', 'rb')
interview_18 = pjct.read_doc_file_to_string('interviews_new/Danijel.docx', 'rb')
interview_19 = pjct.read_doc_file_to_string('interviews_new/Darko.docx', 'rb')
interview_20 = pjct.read_doc_file_to_string('interviews_new/Jusuf.docx', 'rb')
interview_21 = pjct.read_doc_file_to_string('interviews_new/Kenan.docx', 'rb')

interview_list = [interview_1, interview_2, interview_5, interview_6, interview_7,
                  interview_9, interview_10, interview_11, interview_12, interview_13,
                  interview_14, interview_15, interview_16,interview_17,interview_18,
                  interview_19,interview_20,interview_21]

#words to ignore
stop_words = set(nltk.corpus.stopwords.words('english'))
stop_words_add_on = {'okay','like','think','hello','feel','uhm','umm','really','something',
                     'yeah','say','want','yes','lot','make','way','well','get','ono','jel',
                     'ima','mhm','always','every','would','know','see','one','maybe','maybe',
                     'also','said','much','going','little','tell','even','day','thing','quite'
                     'gora','things','people','still','could','show'}
stop_words = stop_words | stop_words_add_on

#creating document collection
interview_tokens = []
j=0
for interview in interview_list:
    interview = pjct.remove_grouped_text(interview)
    interview = pjct.remove_punctuation_characters(interview,interviewer=' ',interviewee=' ')
    interview = interview.lower()
    interview_list[j]=interview
    j = j+1
    words = interview.split()
    words = [word.lower() for word in words if word.lower() not in stop_words]
    words = list({word for word in words if len(word)>2})
    interview_tokens.append(words)
    
#creating word-id dictionary and corpus (bag of words)
num_topics = 3
num_words = 30
dictionary = corpora.Dictionary(interview_tokens)
corpus = [dictionary.doc2bow(interview) for interview in interview_tokens]
del(interview)

#LDA model implementation
print('--Begin LDA--\n')
LDA = models.ldamodel.LdaModel(corpus, num_topics = num_topics, id2word=dictionary, passes=30)
LDA.print_topics(num_topics=num_topics, num_words=num_words)
LDA_results = LDA.show_topics(num_topics=num_topics,num_words=num_words)
print('--End LDA--\n')

#HDP model implementation
print('--Begin HDP--\n')
HDP = models.hdpmodel.HdpModel(corpus,id2word=dictionary)
HDP.print_topics(num_topics=num_topics,num_words=num_words)
HDP_results = HDP.show_topics(num_topics=num_topics,num_words=num_words)
print('--End HDP--\n')

#LSI model implementation
print('--Begin LSI--\n')
LSI = models.lsimodel.LsiModel(corpus,num_topics=num_topics,id2word=dictionary)
LSI.print_topics(num_topics=num_topics,num_words=num_words)
LSI_results = LSI.show_topics(num_topics=num_topics,num_words=num_words)
print('--End LSI--\n')

#create document with results...4 topics, 30 wds each for each model.
folder = '/Users/JaredGallegos/Desktop/Project_1948/Interviews/'
results = w.Document()
results.add_heading('LDA model results',0)
for topic in LDA_results:
    results.add_paragraph(str(topic))
results.add_heading('HDP model results',0)
for topic in HDP_results:
    results.add_paragraph(str(topic))
results.add_heading('LSI model results',0)
for topic in HDP_results:
    results.add_paragraph(str(topic))
results.save(folder+'Topic_model_results.docx')