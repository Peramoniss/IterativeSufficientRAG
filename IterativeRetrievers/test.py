import httpx
from openai import OpenAI
import requests

# payload1 = {
#     "query": "Which film came out earlier, Emmy Of Stork'S Nest or Indira Vizha?",
#     "password": "kTOIe4wQQn 8wQ1XH2jS1 MjkZKOMab6 Rp3eMVT34D 1DitCzvPEm",
#     "documents": [{"title": "Emmy of Sotrk's nest", "text": "text"}, {"title": "Emmy of Sotrk's nest", "text": "Indira Vizha"}, {"title": "Emmy of Sotrk's nest", "text": "STORKS"}, {"title": "Emmy of Sotrk's nest", "text": "Indiana Jones"}]
# }

# payload2 = {
#     "query": "When was Emmy Of Stork'S Nest released?",
#     "password": "kTOIe4wQQn 8wQ1XH2jS1 MjkZKOMab6 Rp3eMVT34D 1DitCzvPEm",
#     "k": 5,
#     "return_corpus_doc": True
# }

# payload3 = {
#     "query": "When was Indira Vizha released?",
#     "password": "kTOIe4wQQn 8wQ1XH2jS1 MjkZKOMab6 Rp3eMVT34D 1DitCzvPEm",
#     "k": 5,
#     "return_corpus_doc": True
# }

# payload4 = {
#     "query": "Emmy Of Stork'S Nest",
#     "password": "kTOIe4wQQn 8wQ1XH2jS1 MjkZKOMab6 Rp3eMVT34D 1DitCzvPEm",
#     "k": 5,
#     "return_corpus_doc": True
# }

# payload5 = {
#     "query": "Indira Vizha",
#     "password": "kTOIe4wQQn 8wQ1XH2jS1 MjkZKOMab6 Rp3eMVT34D 1DitCzvPEm",
#     "k": 5,
#     "return_corpus_doc": True
# }

# response = requests.post("https://renewed-lucid-jiffy.ngrok-free.dev/rerank", json=payload1)
# print("Query 1:", payload1["query"])
# print(response.json())

# response = requests.post("https://16fc307c99e67e.lhr.life/retrieve", json=payload2)
# print("Query 2:", payload2["query"])
# print(response.json())

# response = requests.post("https://16fc307c99e67e.lhr.life/retrieve", json=payload3)
# print("Query 3:", payload3["query"])
# print(response.json())

# response = requests.post("https://16fc307c99e67e.lhr.life/retrieve", json=payload4)
# print("Query 4:", payload4["query"])
# print(response.json())

# response = requests.post("https://16fc307c99e67e.lhr.life/retrieve", json=payload5)
# print("Query 5:", payload5["query"])
# print(response.json())


# messages = [{'role': 'user', 'content': '\n        Q: When did the director of the film Hypocrite die?\n        A: The film Hypocrite was directed by Miguel Morayta. Miguel Morayta died on 19 June 2013. So the answer is: 19 June 2013.\n\n        Q: Do the director of the film Coolie No. 1 (1995 film) and the director of the film The Sensational Trial have the same nationality?   \n        A: Coolie No. 1 (1995 film) was directed by David Dhawan. The Sensational Trial was directed by Karl Freund. David Dhawan\'s nationality is India. Karl Freund\'s nationality is Germany. Thus, they do not have the same nationality. So the answer is: no.   \n\n        Q: Are both Kurram Garhi and Trojkrsti located in the same country?   \n        A: Kurram Garhi is located in the country of Pakistan. Trojkrsti is located in the country of Republic of Macedonia. Thus, they are not in the same country. So the answer is: no.   \n\n        Q: Who was born first out of Martin Hodge and Ivania Martinich?\n        A: Martin Hodge was born on 4 February 1959. Ivania Martinich was born on 25 July 1995. Thus, Martin Hodge was born first. So the answer is: Martin Hodge.\n\n        Q: Which film came out first, The Night Of Tricks or The Genealogy?\n        A: The Night of Tricks was published in the year 1939. The Genealogy was published in the year 1979. Thus, The Night of Tricks came out first. So the answer is: The Night Of Tricks.\n\n        Q: When did the director of the film Laughter In Hell die?\n        A: The film Laughter In Hell was directed by Edward L. Cahn. Edward L. Cahn died on August 25, 1963. So the answer is: August 25, 1963.\n\n        Q: Which film has the director who died later, The Gal Who Took the West or Twenty Plus Two?\n        A: The film Twenty Plus Two was directed by Joseph M. Newman. The Gal Who Took the West was directed by Frederick de Cordova. Joseph M. Newman died on January 23, 2006. Fred de Cordova died on September 15, 2001. Thus, the director who died later is from Twenty Plus Two. So the answer is: Twenty Plus Two.\n        Reference:\nWikipedia Title: "Tarkus (song)"\nFates"" under the title ""Tarkus – Concertante"". Tarkus (song) ""Tarkus"" is the title track of Emerson, Lake & Palmer\'s second album. The progressive-rock epic lasts 20:35. It was the longest studio song by the band until the three impressions of ""Karn Evil 9"". The name ""Tarkus"" refers to the armadillo-tank from the William Neal paintings on the album cover. The artist has explained that the name is an amalgamation between \'Tartarus\' (a place of punishment mentioned in 2 Peter 2:4) and \'carcass\' (hence the name being written in bones on the album cover). Consequently, the name refers to the ""futility\n\n\n\nWikipedia Title: "Tarkus (song)"\nTarkus (song) ""Tarkus"" is the title track of Emerson, Lake & Palmer\'s second album. The progressive-rock epic lasts 20:35. It was the longest studio song by the band until the three impressions of ""Karn Evil 9"". The name ""Tarkus"" refers to the armadillo-tank from the William Neal paintings on the album cover. The artist has explained that the name is an amalgamation between \'Tartarus\' (a place of punishment mentioned in 2 Peter 2:4) and \'carcass\' (hence the name being written in bones on the album cover). Consequently, the name refers to the ""futility of war, a man made mess with\n\n\n\nWikipedia Title: "Tarkus (song)"\nsymbols of mutated destruction."" The song ""Tarkus"" itself supposedly follows the adventures of Tarkus from his birth, through a fight with a manticore, which he loses and concludes with an aquatic version of Tarkus named ""Aquatarkus"". Keith Emerson, when asked what work he is proudest of, named his Piano Concerto (from the ""Works"" release) and Tarkus. ""Tarkus"" itself is broken down into seven parts (timings reflect the beginning cues for each section): The parts supposedly follow Tarkus\' birth in a volcanic eruption centuries before known history. The exact nature of Tarkus\' origin and actions are unknown, and left ambiguous by\n\n\n\nWikipedia Title: "Epitaph (song)"\nincorporate an excerpt from this song after the ""Battlefield"" portion of the live version of their song ""Tarkus"", from the ""Tarkus"" album, as documented in the live album ""Welcome Back My Friends to the Show That Never Ends... Ladies and Gentlemen"". ""Stripes"" from Cage\'s album ""Hell\'s Winter"" samples a middle part of the song throughout its duration. Epitaph Records also took its name from the song. The song reflects the fears of the time, in particular a nuclear apocalypse, and the confusion and despair of the narrator as a result. In 1976, ""Epitaph"" was released as a single with ""21st\n\n\n\nWikipedia Title: "Tarkus (song)"\nit means, but says it is about, ""listening, understanding, hearing"". When asked how Tarkus could be written so quickly (six days), Emerson said: Our sort of creativity comes in varying periods. We get long periods when there isn\'t any creativity, we go into a studio and nothing sounds right, you know. Tarkus was written in six days because there was an awful lot of inspiration and one idea triggered another idea, and it was a long series of ideas being triggered off of what we had already done. The song, described by one journalist to be ""about a post-apocalyptic, metal-plated,\n\n\n\nQ: Answer the following question by concisely reasoning step-by-step (follow the examples above). Where was the place of death of the composer of song Tarkus (Song)?\nA: '}]
# http_client = httpx.Client(verify='petrobras-ca-root.pem')
# if http_client == None:
#     http_client = httpx.Client(timeout=httpx.Timeout(120))
# else:
#     http_client.timeout = httpx.Timeout(120)
# client = OpenAI(
#         api_key="d37099ddd8094dfc8fe5c2ebf220532d",
#         base_url="https://apit.petrobras.com.br/ia/texto/v1/litellm/litellm/",
#         default_headers={
#             "api-key": "d37099ddd8094dfc8fe5c2ebf220532d"  # Matches your HTTP snippet header
#         },
#         http_client=http_client 
#     )
# for i in range(100):
#     response = client.chat.completions.create(
#         model="deepseek-v4-flash",
#         messages=messages,
#         temperature=0.0000000000,
#         max_tokens=2000,
#         # top_p=0.0001,
#         frequency_penalty=0,
#         presence_penalty=0,
#         seed = 42,
#         response_format= None,
#         moderation=None,
#         reasoning_effort='none'
#     )
#     usage = response.usage

#     answer = response.choices[0].message.content
#     print(answer)


# import nltk

# # Download the Punkt tokenizer model (only needed once)
# nltk.download('punkt')

# from nltk.tokenize import sent_tokenize

# text = "Dr. Martin L. King Jr. is an activist. I mean, was. Price is 3.14 dollars... Really."

# for s in sent_tokenize(text):
#     print(s)

from Inference.requestManagers import OllamaRequestManager
request_manager = OllamaRequestManager(api_key="sk-c1d67e5489424c3a8373dd96e9377407", timeout=120) #https://ollama.k8s.inf.ufrgs.br/ollama
response = request_manager.query_model([{'role': 'user', 'content': 'Oi Cueio. Tudo bem'}], model='mistral-7b-instruct')
print(response)