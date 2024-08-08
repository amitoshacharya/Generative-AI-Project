# NOTES

## ***[Word Embeddings](https://www.geeksforgeeks.org/word-embeddings-in-nlp/)***
- Word Embeddings is a way to represent words or documents in a form of numerical matrices called as **Vectors**.
- Word Embedding or Word Vector is a numeric vector/matrix input that represents a word in a *lower-dimensional space*. It allows words with similar meanings to have a similar representation.
- Word Embeddings are a method of extracting features out of text so that we can input those features into a machine learning model to work with text data. 
- They try to preserve ***syntactical*** and ***semantic*** information.
- Word Embeddings provide solutions to the problems with vectorization methods.
    + Vectorization Methods:
        + Bag of Words (BOW)
        + CountVectorizer  
        + TFIDF

    + Problems with Vectorization Methods:
        + These Vectorization methods rely on the word count in a sentence but do not save any syntactical or semantic information. 
        + In these algorithms, the size of the vector is the number of elements in the vocabulary. 
        + We can get a sparse matrix if most of the elements are zero. 
        + ***Large input vectors will mean a huge number of weights which will result in high computation required for training.***

- Need for Word Embedding?
    + To reduce dimensionality.
        + E.g.: happy --> [0, 1, 0, 1]
    + To use a word to predict the words around it.
        + E.g.: 
        Like --> [0, 1, 0, 1]
        Love --> [0, 1, 0, 1]
        
    + Inter-word semantics must be captured.
        + E.g.: 
        I Like you--> 
        [ [1,0,1,0], [0, 1, 0, 1], [1, 0.7, 1, 0.3] ] 
        We Love India--> 
        [ [1,0,0,0], [0, 1, 0, 1], [1, 0.5, 1, 0.6] ]

- How are Word Embeddings used?
    + They are used as input to machine learning models.
    
        Take the words —-> Give their numeric representation —-> Use in training or inference.
        
    + To represent or visualize any underlying patterns of usage in the corpus that was used to train them.

- [Word Embeddings Methods](https://www.geeksforgeeks.org/word-embeddings-in-nlp/)
    + Neural Approach
        + Word2Vec
        + Continuous Bag of Words(CBOW)
        + Skip-Gram
    + Pre-trained Word-Embedding
        + GloVe
        + Fasttext
        + BERT (Bidirectional Encoder Representations from Transformers)

### ***Frequently Asked Questions (FAQs) related to Embeddings***
1. Does GPT use word embeddings?
    - GPT uses context-based embeddings rather than traditional word embeddings. 
    - It captures word meaning in the context of the entire sentence.

2. What is the difference between BERT and word embeddings?
    - BERT is contextually aware, considering the entire sentence, while traditional word embeddings, like Word2Vec, treat each word independently.

3. What are the two types of word embedding?
    - Word embeddings can be broadly evaluated in two categories, [intrinsic and extrinsic](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/EDF43F837150B94E71DBB36B28B85E79/S204877031900012Xa.pdf/div-class-title-evaluating-word-embedding-models-methods-and-experimental-results-div.pdf). 
    - For intrinsic evaluation, word embeddings are used to calculate or predict semantic similarity between words, terms, or sentences.
    - For extrinsic evaluation, word embeddings are used as input features to a downstream task and measure changes in performance metrics specific to that task. 
    Five extrinsic evaluators: (1) POS tagging, (2) chunking, (3) named-entity recognition, (4) sentiment analysis, and (5) neural machine translation (NMT).

4. How does word vectorization work?
    - Word vectorization converts words into numerical vectors, capturing semantic relationships. 
    - Techniques like TF-IDF, Word2Vec, and GloVe are common.

5. What are the benefits of word embeddings?
    - Word embeddings offer semantic understanding, capture context, and enhance NLP tasks. 
    - They reduce dimensionality, speed up training, and aid in language pattern recognition.

## [Generative Pre-Trained Transformer (GPT)](https://aws.amazon.com/what-is/gpt/#seo-faq-pairs#how-does-gpt-work)
- The GPT models are neural network-based language prediction models built on the Transformer architecture. They analyze natural language queries, known as ***prompts***, and predict the best possible response based on their understanding of language.

- **Working Process** 
    + To do that, the GPT models rely on the knowledge they gain after they're trained with hundreds of billions of parameters on massive language datasets. 
    + They can take input context into account and dynamically attend to different parts of the input, making them capable of generating long responses, not just the next word in a sequence. 
        + *For example*, when asked to generate a piece of Shakespeare-inspired content, a GPT model does so by remembering and reconstructing new phrases and entire sentences with a similar literary style.

    + There are different types of neural networks, like recurrent and convolutional. 
        + The ***GPT models are transformer neural networks***. 
        + The transformer neural network architecture uses self-attention mechanisms to focus on different parts of the input text during each processing step. 
        + A transformer model captures more context and improves performance on natural language processing (NLP) tasks. 
        + It has two main modules:
            + Encoder
            + Decoder
    
- ***Encoder***
    + *Transformers pre-process text inputs as embeddings, which are mathematical representations of a word. When encoded in vector space, words that are closer together are expected to be closer in meaning.*
    + These word embeddings are processed through an ***encoder*** component that ***captures contextual information from an input sequence (or from word embeddings)***. 
    + When it receives input, the transformer network's encoder block separates words into embeddings and assigns weight to each. 
    + Weights are parameters to indicate the relevance/importance of words in a sentence.

    + Additionally, position encoders allow GPT models to prevent ambiguous meanings when a word is used in other parts of a sentence. 
        + *For example*, position encoding allows the transformer model to differentiate the semantic differences between these sentences: 
           + A dog chases a cat
           + A cat chases a dog

    + So, the encoder processes the input sentence and generates a fixed-length vector representation, known as an embedding. This representation is used by the decoder module.

- ***Decoder***
    + The decoder uses the vector representation to predict the requested output. 
    + ***It has built-in self-attention mechanisms to focus on different parts of the input and guess the matching output***. 
    + Complex mathematical techniques help the decoder to estimate several different outputs and predict the most accurate one.
    + Compared to its predecessors, like recurrent neural networks, transformers are more parallelizable because they do not process words sequentially one at a time, but instead, process the entire input all at once during the learning cycle. 
    + Due to this, engineers spent the thousands of hours fine-tuning and training the GPT models, they're able to give fluent answers to almost any provided input.

## [Large Language Models (LLM)](https://aws.amazon.com/what-is/large-language-model/)
- LLMs are very large deep learning models that are pre-trained on vast amounts of data. 
- The underlying transformer is a set of neural networks that consist of an encoder and a decoder with self-attention capabilities. 
- The encoder and decoder extract meanings from a sequence of text and understand the relationships between words and phrases in it.

- ***Difference Between LLMs (Transformers) and RNNs (predecessors of Transformers)***
    + Transformer LLMs are capable of unsupervised training, although a more precise explanation is that transformers perform self-learning. 
    + It is through this process that transformers learn to understand basic grammar, languages, and knowledge.
    + Unlike earlier recurrent neural networks (RNN) that sequentially process inputs, transformers process entire sequences in parallel. 
    + This allows the data scientists to use GPUs for training transformer-based LLMs, significantly reducing the training time.

- ***Benefits of Transformer LLMs***
    + Transformer neural network architecture allows the use of very large models, often with hundreds of billions of parameters. 
    + Such large-scale models can ingest massive amounts of data, often from the internet, but also from sources such as the web crawling, which comprises more than 50 billion web pages, and Wikipedia, which has approximately 57 million pages.

- **Work Process**
    + A key factor in how LLMs work is the way they represent words. 
    + Earlier forms of machine learning used a numerical table to represent each word. But, this form of representation could not recognize relationships between words such as words with similar meanings. 
    + This limitation was overcome by using multi-dimensional vectors, commonly referred to as word embeddings or word vectors, to represent words so that words with similar contextual meanings or other relationships are close to each other in the vector space.

    + Using word embeddings, transformers can pre-process text as numerical representations through the encoder and understand the context of words and phrases with similar meanings as well as other relationships between words such as parts of speech. 
    + It is then possible for LLMs to apply this knowledge of the language through the decoder to produce a unique output.

    + Word Embeddings --> Encoder --> Decoder --> Output

- ***How are large language models trained?*** <img src="https://miro.medium.com/v2/resize:fit:1100/format:webp/1*aNf8qJHyrd8zGE199E387g.png">
    + **Pre-training**
        + The first stage is pre-training, this stage requires massive amounts of data to learn to predict the next word. 
        + In this phase, the model learns not only to master the grammar and syntax of language, but it also acquires a great deal of knowledge about the world, and even some other emerging abilities such as text summarization, Q & A etc.
        + ***What might be the problem with this kind of pre-training?***
            + Well, there are certainly a few, but the one important point is with what the LLM has really learned.
            + Namely, it has learned mainly to ramble on about a topic. 
            + It may even be doing an incredibly good job, but what it doesn't do is respond well to the kind of inputs we would generally expect from AI, such as a question or an instruction.
            + The problem is that this model has not learned to be, and so is not behaving as, an assistant.
            + *For example*,
             If you ask a pre-trained LLM "What is your fist name?" it may respond with "What is your last name?" simply because this is the kind of data it has seen during pre-training, as in many empty forms. It's only trying to complete the input sequence.

        + At this stage, the LLM is not aligned with human intentions.  
        + Alignment is an important topic for LLMs. So even though initially they don't respond well to instructions, they can be taught to do so.

    + **Instruction fine-tuning**
        + This is where instruction tuning comes in. We take the pre-trained LLM with its current abilities and do essentially what we did before — i.e., learn to predict one word at a time — but now we do this using only high-quality instruction and response pairs as our training data.
        + That way, the model un-learns to simply be a text completer and learns to become a helpful assistant that follows instructions and responds in a way that is aligned with the user's intention. 
        + The size of this instruction dataset is typically a lot smaller than the pre-training set. This is because the high-quality instruction-response pairs are much more expensive to create as they are typically sourced from humans. 
        + This is very different from the inexpensive self-supervised labels we used in pre-training. This is why this stage is also called ***supervised instruction fine-tuning***.
    
    + **RLHF**
        + There is also a third stage that some LLMs like ChatGPT go through, which is **reinforcement learning from human feedback (RLHF)**. 
        + It's purpose is similar to instruction fine-tuning. RLHF also helps alignment and ensures that the LLM's output reflects human values and preferences. 
        + There is some early research that indicates that this stage is critical for reaching or surpassing human-level performance. 
        + In fact, combining the fields of reinforcement learning and language modeling is being shown to be especially promising and is likely to lead to some massive improvements over the LLMs we currently have.

- Three common learning models exist:
    + ***Zero-shot learning*** 
        + Base LLMs can respond to a broad range of requests without explicit training, often through prompts, although answer accuracy varies.

    - ***Few-shot learning***
        + By providing a few relevant training examples, base model performance significantly improves in that specific area.

    - ***Fine-tuning***
        + This is an extension of few-shot learning and a process that leverages transfer learning to 'adapt' the model to a downstream task or to solve a specific problem. 
        + Differently from few-shot learning and RAG, it results in a new model being generated, with updated weights and biases. 
        + It requires a set of training examples consisting of a single input (the prompt) and its associated output (the completion). 
        + This would be the preferred approach if:
            + `Using fine-tuned models.`
                + A business would like to use fine-tuned less capable models (like embedding models) rather than high performance models, resulting in a more cost effective and fast solution.

            + `Considering latency.`
                + Latency is important for a specific use-case, so it's not possible to use very long prompts or the number of examples that should be learned from the model doesn't fit with the prompt length limit.

            + `Staying up to date.`
                + A business has a lot of high-quality data and ground truth labels and the resources required to maintain this data up to date over time.

- [Difference Between Large Language Models and Generative AI](https://www.analyticsvidhya.com/blog/2023/03/an-introduction-to-large-language-models-llms/)

## [LangChain](https://python.langchain.com/v0.2/docs/introduction/) 
- LangChain is a framework designed to simplify the creation of applications using large language models (LLMs). 
- It provides a standard interface for chains, integrates with other tools, and offers end-to-end solutions for common applications. 
- Essentially, it's a library of abstractions for Python and JavaScript, representing common steps and concepts necessary for working with language models. 
- Whether building ChatBots, document analysis tools, or code analyzers, LangChain can help streamline the LLM-powered development process. 
- In context-aware, reasoning applications, LangChain is worth exploring.
- The framework consists of the following open-source libraries:
    + ***langchain-core***: Base abstractions and LangChain Expression Language.
    + ***langchain-community***: Third party integrations.
        + ***Partner packages*** *(e.g. **langchain-openai**, **langchain-anthropic**, etc.)*: Some integrations have been further split into their own lightweight packages that only depend on langchain-core.
    + ***langchain***: Chains, agents, and retrieval strategies that make up an application's cognitive architecture.

- Components under LangChain - [Click Here](https://python.langchain.com/v0.2/docs/integrations/components/)

## [Retrieval-Augmented Generation (RAG)](https://aws.amazon.com/what-is/retrieval-augmented-generation/)
- Retrieval-Augmented Generation (RAG) is the ***process of optimizing the output of a large language model***, so it references an authoritative knowledge base outside of its training data sources before generating a response.
- Large Language Models (LLMs) are trained on vast volumes of data and use billions of parameters to generate original output for tasks like answering questions, translating languages, and completing sentences. 
- RAG extends the already powerful capabilities of LLMs to specific domains or an organization's internal knowledge base, all without the need to retrain the model. 
- It is a cost-effective approach to improving LLM output so it remains relevant, accurate, and useful in various contexts.
- **Why is RAG important?**
    + LLMs are a key artificial intelligence (AI) technology powering intelligent ChatBots and other natural language processing (NLP) applications. 
    + The goal is to create bots that can answer user questions in various contexts by cross-referencing authoritative knowledge sources. 
    + Unfortunately, the nature of LLM technology introduces unpredictability in LLM responses. 
    + Additionally, LLM training data is static and introduces a cut-off date on the knowledge it has.
    + ***Known challenges of LLMs include:***
        + Presenting false information when it does not have the answer. ***(Hallucination)***
        + Presenting out-of-date or generic information when the user expects a specific, current response.
        + Creating a response from non-authoritative sources.
        + Creating inaccurate responses due to terminology confusion, wherein different training sources use the same terminology to talk about different things.

    + ***RAG is one approach to solving some of these challenges.*** 
        + It redirects the LLM to retrieve relevant information from authoritative, pre-determined knowledge sources. 
        + Organizations have greater control over the generated text output, and users gain insights into how the LLM generates the response.

- **What are the benefits of RAG?**
    + ***Cost-effective implementation***
        + Chatbot development typically begins using a foundation model. Foundation models (FMs) are API-accessible LLMs trained on a broad spectrum of generalized and unlabeled data. 
        + The computational and financial costs of fine-tuning FMs (i.e.; retraining FMs for organization or domain-specific information) are high. 
        + RAG is a more cost-effective approach to introducing new data to the LLM. 
        + It makes generative AI technology more broadly accessible and usable.

    + ***Current information***
        + Even if the original training data sources for an LLM are suitable for your needs, it is challenging to maintain relevancy. 
        + RAG allows developers to provide the latest research, statistics, or news to the generative models. 
        + They can use RAG to connect the LLM directly to live social media feeds, news sites, or other frequently-updated information sources. 
        + The LLM can then provide the latest information to the users.

    + ***Enhanced user trust***
        + RAG allows the LLM to present accurate information with source attribution. The output can include citations or references to sources. 
        + Users can also look up source documents themselves if they require further clarification or more detail. 
        + This can increase trust and confidence in there generative AI solution.

    + ***More developer control***
        + With RAG, developers can test and improve their chat applications more efficiently. 
        + They can control and change the LLM's information sources to adapt to changing requirements or cross-functional usage. 
        + Developers can also restrict sensitive information retrieval to different authorization levels and ensure the LLM generates appropriate responses. 
        + In addition, they can also troubleshoot and make fixes if the LLM references incorrect information sources for specific questions. 
        + Organizations can implement generative AI technology more confidently for a broader range of applications.

- **How does Retrieval-Augmented Generation (RAG) work?**
    + ***With and Without RAG***
        + Without RAG, the LLM takes the user input and creates a response based on information it was trained on—or what it already knows. 
        + With RAG, an information retrieval component is introduced that utilizes the user input to first pull information from a new data source. 
        + The user query and the relevant information are both given to the LLM. The LLM uses the new knowledge and its training data to create better responses. 

    + *The following sections provide an overview of the process:*
        + ***Create external data***
            + The new data outside of the LLM's original training data set is called ***external data***. 
            + It can come from multiple data sources, such as a APIs, databases, or document repositories. 
            + The data may exist in various formats like files, database records, or long-form text. 
            + Another AI technique, called ***embedding language models***, converts data into numerical representations and stores it in a vector database. This process creates a knowledge library that the generative AI models can understand.

        + ***Retrieve relevant information***
            + The next step is to perform a relevancy search. 
            + The user query is converted to a vector representation and matched with the vector databases. 
                + *For example*, 
                Consider a smart chatbot that can answer human resource questions for an organization. 
                If an employee searches, "How much annual leave do I have?" the system will retrieve annual leave policy documents alongside the individual employee's past leave record. 
                These specific documents will be returned because they are highly-relevant to what the employee has input. The relevancy was calculated and established using mathematical vector calculations and representations.

        + ***Augment the LLM prompt***
            + Next, the RAG model augments the user input (or prompts) by adding the relevant retrieved data in ***context***. 
            + This step uses prompt engineering techniques to communicate effectively with the LLM. 
            + The augmented prompt allows the large language models to generate an accurate answer to user queries.

        + ***Update external data***
            + The next question may be - what if the external data becomes stale or out-dated ? To maintain current information for retrieval, asynchronously update the documents and update embedding representation of the documents. 
            + This can be done through automated real-time processes or periodic batch processing. 
            + This is a common challenge in data analytics. Thus, different data-science approaches to change management can be used.

    + *The following diagram shows the conceptual flow of using RAG with LLMs.* <img src="https://docs.aws.amazon.com/images/sagemaker/latest/dg/images/jumpstart/jumpstart-fm-rag.jpg">

- **Difference between Conventional and Semantic search in RAG**
    + *Semantic search enhances RAG results for organizations wanting to add vast external knowledge sources to their LLM applications.* 
        + Modern enterprises store vast amounts of information like manuals, FAQs, research reports, customer service guides, and human resource document repositories across various systems.
        + Context retrieval is challenging at scale and consequently lowers generative output quality.

    + *Semantic search technologies can scan large databases of disparate information and retrieve data more accurately*. 
        + *For example*, 
        They can answer questions such as, "How much was spent on machinery repairs last year?" by mapping the question to the relevant documents and returning specific text instead of search results. 
        Developers can then use that answer to provide more context to the LLM.

    + ***Difference*** 
        + Conventional or keyword search solutions in RAG produce limited results for knowledge-intensive tasks. Developers must also deal with word embeddings, document chunking, and other complexities as they manually prepare their data. 

        + In contrast, semantic search technologies do all the work of knowledge base preparation so developers don't have to. They also generate semantically relevant passages and token words ordered by relevance to maximize the quality of the RAG payload.


## RAG Mechanism Steps

- RAG is a technique that improves the capabilities of LLMs by combining them with external data sources. 
- There are three key functionalities of RAG.
    + **Retrieval** 
    This is where the most relevant information is identified(from the external data).

    + **Augmentation** 
    This step involves preprocessing and analyzing the retrieved information to make it suitable for the LLM using Summarization, Fact Checking and Formatting.

    + **Generation** 
    With the retrieved and augmented information at end, the LLM generates better in context responses.

- [Retrieval](https://python.langchain.com/v0.1/docs/modules/data_connection/)<img src="https://python.langchain.com/v0.1/assets/images/data_connection-95ff2033a8faa5f3ba41376c0f6dd32a.jpg" width=auto>

    + [Document loaders](https://python.langchain.com/v0.1/docs/modules/data_connection/document_loaders/)<img src="https://miro.medium.com/v2/resize:fit:828/format:webp/1*UBX8w3Ef-89P-Ezv57L1hw.png" width=auto>

        + Document loaders are tools that play a crucial role in data ingestion. 
        + They take in raw data from different sources and convert them into a structured format called "Documents". 
        + These documents contain the document content as well as the associated metadata like source and timestamps. 
        + *LangChain* provides over 100 different document loaders as well as integrations with other major providers in the space, like AirByte and Unstructured. 
        + LangChain provides integrations to load all types of documents (HTML, PDF, code) from all types of locations (private S3 buckets, public websites).
        + ***Package Name:*** *langchain_community.document_loaders* or *langchain_text_splitters*

    + [Text Splitting](https://medium.com/@sushmithabhanu24/retrieval-in-langchain-part-2-text-splitters-2d8c9d595cc9)
        + Any NLP task involving a long document might need to be preprocessed or transformed to improve the accuracy or efficiency of the task at hand. 
        + Text Splitter comes in handy when it comes to breaking down huge documents into chunks that will enable analysis at a more granular level. 
        + LangChain provides the user with various options to transform the documents by chunking them into meaningful portions and then combining the smaller chunks into larger chunks of a particular size with overlap to retain the context.
        + ***Package Name:*** *langchain.text_splitter*
        + **Types of Text Splitters**
            + [*Character Text Splitter*](https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/character_text_splitter/)
                + This is the simplest method of splitting the text by characters which is computationally cheap and doesn't require the use of any NLP libraries. 
                + Here the text split is done on characters (by default "nn") and the chunk size is measured by the number of characters.
                + The parameter chunk_overlap helps in retaining the semantic context between the chunks. 
                + The metadata can also be passed along with the documents.
                + **API Reference:**
                    + CharacterTextSplitter
            
            + [*Recursive Character Text Splitter*](https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/recursive_text_splitter/)
                + This text splitter is the recommended one for generic text. 
                + It is parameterized by a list of characters. 
                + It tries to split on them in order until the chunks are small enough. 
                + The default list is ["nn", "n", " ", ""]. This has the effect of trying to keep all paragraphs (and then sentences, and then words) together as long as possible, as those would generically seem to be the strongest semantically related pieces of text.
                + Here the text split is done on the list of characters and the chunk size is measured by the number of characters.
                + **API Reference:**
                    + RecursiveCharacterTextSplitter

            + [*RecursiveJsonSplitter*](https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/recursive_json_splitter/)
                + This json splitter traverses json data depth first and builds smaller json chunks. 
                + It attempts to keep nested json objects whole but will split them if needed to keep chunks between a min_chunk_size and the max_chunk_size. 
                + If the value is not a nested json, but rather a very large string the string will not be split. If you need a hard cap on the chunk size consider Recursive Text splitter on those chunks. 
                + There is an optional pre-processing step to split lists, by first converting them to json (dict) and then splitting them as such.
                + How the text is split on json value and the chunk size is measured by number of characters.
                + **API Reference:**
                    + RecursiveJsonSplitter

            + [*Code Splitter*](https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/code_splitter/)
                + This type lets you split the code and it comes with multiple language options like Python, java, Latex, HTML, scala, c, and a lot more.
                +  Import enum *Language* and specify the language.
                + **API Reference:**
                    + Language
                    + RecursiveCharacterTextSplitter 

            + [*SemanticChunker*](https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/semantic-chunker/)  
                + Semantic chunking involves taking the embeddings of every sentence in the document, comparing the similarity of all sentences with each other, and then grouping sentences with the most similar embeddings together.
                + By focusing on the text's meaning and context, Semantic Chunking significantly enhances the quality of retrieval. 
                + It's a top-notch choice when maintaining the semantic integrity of the text is vital.
                + There are 3 different strategies of Semantic Chunking and can be used with breakpoint_threshold_type:
                    + `percentile` (default) 
                    In this method, all differences between sentences are calculated, and then any difference greater than the X percentile is split.

                    + `standard_deviation` 
                    In this method, any difference greater than X standard deviations is split.

                    + `interquartile` 
                    In this method, the interquartile distance is used to split chunks.
                + **API Reference:**
                    + SemanticChunker
                    + OpenAIEmbeddings

    + [Text embedding models](https://python.langchain.com/v0.1/docs/modules/data_connection/text_embedding/)
        + The Embeddings class is a class designed for interfacing with text embedding models. 
        + There are lots of embedding model providers (OpenAI, Cohere, Hugging Face, etc) - this class is designed to provide a standard interface for all of them.

        + Embeddings create a vector representation of a piece of text. This is useful because it means we can think about text in the vector space, and do things like semantic search where we look for pieces of text that are most similar in the vector space.

        + The base Embeddings class in LangChain provides two methods: 
            + one for embedding documents and 
            + one for embedding a query. 
        + The former takes as input multiple texts, while the latter takes a single text. 
        + The reason for having these as two separate methods is that some embedding providers have different embedding methods for documents (to be searched over) vs queries (the search query itself).
    
    + [Vector Store](https://python.langchain.com/v0.2/docs/how_to/vectorstores/)<img src="https://python.langchain.com/v0.1/assets/images/vector_stores-125d1675d58cfb46ce9054c9019fea72.jpg">

        + One of the most common ways to store and search over unstructured data is to embed it and store the resulting embedding vectors, and then at query time to embed the unstructured query and retrieve the embedding vectors that are 'most similar' to the embedded query. 
        + A vector store takes care of storing embedded data and performing vector search.
        + A key part of working with vector stores is creating the vector to put in them, which is usually created via embeddings. 
        + There are many great vector store options, here are a few that are free, open-source, and run entirely on your local machine. Review all integrations for many great hosted offerings.
            + Chroma 
            ***Package Name:*** *langchain_chroma*

            + Facebook AI Similarity Search (FAISS) 
            ***Package Name:*** *langchain_community.vectorstores*  

            + Lance 
            ***Package Name:*** *langchain_community.vectorstores* and *lancedb*

    + [Retriever](https://python.langchain.com/v0.2/docs/how_to/vectorstore_retriever/)
        + A vector store retriever is a retriever that uses a vector store to retrieve documents. 
        + It is a lightweight wrapper around the vector store class to make it conform to the retriever interface. 
        + It uses the search methods implemented by a vector store, like similarity search and MMR, to query the texts in the vector store.
        + Steps to work with Retriever:
            + *Instantiate a retriever from a vector store* 
            To build a retriever from a vector store using its ***.as_retriever*** method. 

            + *Specify the search type for the retriever* 
                + By default, the vector store retriever uses similarity search. 
                + If the underlying vector store supports maximum marginal relevance (mmr) search, specify that as the search type.
                + This effectively specifies the method on the underlying vector store is used (e.g., similarity_search, max_marginal_relevance_search, etc.).
                
            + *Specify additional search parameters, such as threshold scores and top-k*
                + We can pass parameters to the underlying vector-store's search methods using search_kwargs
                + ***Similarity score threshold retrieval***
                    + we can set a similarity score threshold. 
                    + only return documents with a score above that threshold.
                + ***Specifying top k*** 
                The number of documents can be limit up to **k** returned by the retriever.

            + *Example:*
                + **retriever** = **vectorstore.as_retriever**(**search_type=**"similarity_score_threshold", **search_kwargs=**{**"score_threshold":** 0.5, **"k":** 1
                })

#### ***Differences between different Search Types***
- ***Maximal Marginal Relevance (MMR):***
    + **Purpose:** MMR aims to select diverse examples while optimizing for similarity to the input.
    + **How it works:**
        + It identifies examples with embeddings (representations) that have high cosine similarity to the input.
        + It iteratively adds examples, penalizing them for closeness to already selected ones.
    + **Advantages:**
        + Balances relevance and diversity.
        + Useful for tasks like few-shot learning.
    + **Example:** If given "worried," MMR might select "happy" as the antonym1.

- ***Similarity:***
    + **Purpose:** Retrieve relevant examples from a vector store (embedding database).
    + **How they work:**
        + Use similarity search methods (e.g., cosine similarity) to query embeddings.
        + Can incorporate MMR.
    + **Advantages:**
        + Leverage external knowledge.
        + Improve response quality.
    + **Example:** Retrieving relevant examples for a given task using embeddings3.
    
- In summary, MMR balances relevance and diversity, n-gram overlap captures local context, and similarity retrievers enhance responses by leveraging external information. 

## [Prompt Engineering](https://aws.amazon.com/what-is/prompt-engineering/)
- Prompt engineering is the process where we guide generative AI solutions to generate desired outputs. 
- Even though generative AI attempts to mimic humans, it requires detailed instructions to create high-quality and relevant output. 
- In prompt engineering, we choose the most appropriate formats, phrases, words, and symbols that guide the AI to interact with respective users more meaningfully. 
- Prompt engineers use creativity plus trial and error to create a collection of input texts, so an application's generative AI works as expected.
- **Prompts**
    + A prompt is a natural language text that requests the generative AI to perform a specific task. 
    + GenerativeAI is an AI solution that creates new content like stories, conversations, videos, images, and music. 
    + It's powered by very large machine learning (ML) models that use deep neural networks that have been pre-trained on vast amounts of data.

    + The large language models (LLMs) are very flexible and can perform various tasks. 
    + *For example*, they can summarize documents, complete sentences, answer questions, and translate languages. 
    + For specific user input, the models work by predicting the best output that they determine from past training.

    + However, because they're so open-ended, users can interact with generative AI solutions through countless input data combinations. 
    + The AI language models are very powerful and don't require much to start creating content. Even a single word is sufficient for the system to create a detailed response.

    + That being said, not every type of input generates helpful output. 
    + Generative AI systems require context and detailed information to produce accurate and relevant responses. 
    + When we systematically design prompts, we get more meaningful and usable creations. 
    + In prompt engineering, we continuously refine prompts until we get the desired outcomes from the AI system.

- ***Why is prompt engineering important?***
    + *Greater developer control*
        + Prompt engineering gives developers more control over user's interactions with the AI. + Effective prompts provide intent and establish ***context*** to the large language models (llm). 
        + They help the AI refine the output and present it concisely in the required format.

        + They also prevent your users from misusing the AI or requesting something the AI does not know or cannot handle accurately. 
        + *For instance*, to limit users from generating inappropriate content in a business AI application.

    + *Improved user experience*
        + Users avoid trial and error and still receive coherent, accurate, and relevant responses from AI tools. 
        + Prompt engineering makes it easy for users to obtain relevant results in the first prompt. 
        + It helps mitigate bias that may be present from existing human bias in the large language model's training data.

        + Further, it enhances the user-AI interaction so the AI understands the user's intention even with minimal input. 
        + *For example*, requests to summarize a legal document and a news article get different results adjusted for style and tone. 
        + This is true even if both users just tell the application, "Summarize this document."

    + *Increased flexibility*
        + Higher levels of abstraction improve AI models and allow organizations to create more flexible tools at scale. 
        + A prompt engineer can create prompts with domain-neutral instructions highlighting logical links and broad patterns. 
        + Organizations can rapidly reuse the prompts across the enterprise to expand their AI investments.

        + *For example*, to find opportunities for process optimization, the prompt engineer can create different prompts that train the AI model to find inefficiencies using broad signals rather than context-specific data. The prompts can then be used for diverse processes and business units.

- ***Prompt Engineering use cases***
    + Prompt engineering techniques are used in sophisticated AI systems to improve user experience with the learning language model.

    + Here are some examples:
        + *Subject matter expertise*
            + Prompt engineering plays a key role in applications that require the AI to respond with subject matter expertise. 
            + A prompt engineer with experience in the field can guide the AI to reference the correct sources and frame the answer appropriately based on the question asked.

            + *For example*, 
            In the medical field, a physician could use a prompt-engineered language model to generate differential diagnoses for a complex case. 
            The medical professional only needs to enter the symptoms and patient details. 
            The application uses engineered prompts to guide the AI first to list possible diseases associated with the entered symptoms. Then it narrows down the list based on additional patient information.

        + *Critical thinking*
            + Critical thinking applications require the language model to solve complex problems. 
            + To do so, the model analyzes information from different angles, evaluates its credibility, and makes reasoned decisions. 
            + Prompt engineering enhances a model's data analysis capabilities.

            + *For instance*, 
            In decision-making scenarios, we could prompt a model to list all possible options, evaluate each option, and recommend the best solution.

        + *Creativity*
            + Creativity involves generating new ideas, concepts, or solutions. 
            + Prompt engineering can be used to enhance a model's creative abilities in various scenarios.

            + *For instance*, 
            In writing scenarios, a writer could use a prompt-engineered model to help generate ideas for a story. 
            The writer may prompt the model to list possible characters, settings, and plot points then develop a story with those elements. 
            Or a graphic designer could prompt the model to generate a list of color palettes that evoke a certain emotion then create a design using that palette. 

- [***What are prompt engineering techniques?***](https://github.com/microsoft/generative-ai-for-beginners/tree/main/05-advanced-prompts#techniques-for-prompting)
    + Prompt engineering is a dynamic and evolving field. 
    + It requires both linguistic skills and creative expression to fine-tune prompts and obtain the desired response from the generative AI tools.
    + Here are some more examples of techniques that prompt engineers use to improve their AI models' natural language processing (NLP) tasks:
        + `Zero-shot prompting`
            + Zero-shot prompting provides the machine learning model with a task it hasn't explicitly been trained on. 
            + Zero-shot prompting tests the model's ability to produce relevant outputs without relying on prior examples.

        + `Few-shot prompting`
            + Few-shot prompting or in-context learning gives the model a few sample outputs (shots) to help it learn what the requestor wants it to do. 
            + The learning model can better understand the desired output if it has context to draw on.

        + `Chain-of-thought (COT) prompting`
            + Chain-of-thought prompting is a technique that breaks down a complex question into smaller, logical parts that mimic a train of thought. 
            + This helps the model solve problems in a series of intermediate steps rather than directly answering the question. 
            + This enhances its reasoning ability.
            + You can perform several chain-of-thoughts rollout for complex tasks and choose the most commonly reached conclusion. 
            + If the rollout disagree significantly, a person can be consulted to correct the chain of thought.

            + *For example*, 
            If the question is "What is the capital of France?" the model might perform several rollout leading to answers like "Paris," "The capital of France is Paris," and "Paris is the capital of France." 
            Since all rollout lead to the same conclusion, "Paris" would be selected as the final answer.

        + `Tree-of-thought prompting`
            + The tree-of-thought technique generalizes chain-of-thought prompting. 
            + It prompts the model to generate one or more possible next steps. 
            + Then it runs the model on each possible next step using a tree search method.

            + *For example*, 
            If the question is "What are the effects of climate change?" the model might first generate possible next steps like "List the environmental effects" and "List the social effects." 
            It would then elaborate on each of these in subsequent steps.

        + `Maieutic prompting`
            + Maieutic prompting is similar to tree-of-thought prompting. 
            + The model is prompted to answer a question with an explanation. 
            + The model is then prompted to explain parts of the explanation. 
            + Inconsistent explanation trees are pruned or discarded. 
            + This improves performance on complex commonsense reasoning.

            + *For example*, 
            If the question is "Why is the sky blue?" the model might first answer, "The sky appears blue to the human eye because the short waves of blue light are scattered in all directions by the gases and particles in the Earth's atmosphere." 
            It might then expand on parts of this explanation, such as why blue light is scattered more than other colors and what the Earth's atmosphere is composed of.

        + `Complexity-based prompting`
            + This prompt-engineering technique involves performing several chain-of-thought rollout. 
            + It chooses the rollout with the longest chains of thought then chooses the most commonly reached conclusion.
            + *For example*, 
            If the question is a complex math problem, the model might perform several rollout, each involving multiple steps of calculations. 
            It would consider the rollout with the longest chain of thought, which for this example would be the most steps of calculations. 
            The rollout that reach a common conclusion with other rollout would be selected as the final answer.

        + `Generated knowledge prompting`
            + This technique involves prompting the model to first generate relevant facts needed to complete the prompt. 
            + Then it proceeds to complete the prompt. 
            + This often results in higher completion quality as the model is conditioned on relevant facts.

            + *For example*, 
            Imagine a user prompts the model to write an essay on the effects of deforestation. 
            The model might first generate facts like "deforestation contributes to climate change" and "deforestation leads to loss of biodiversity." 
            Then it would elaborate on the points in the essay.

        + `Least-to-most prompting`
            + In this prompt engineering technique, the model is prompted first to list the sub-problems of a problem, and then solve them in sequence. 
            + This approach ensures that later sub-problems can be solved with the help of answers to previous sub-problems.

            + *For example*, 
            Imagine that a user prompts the model with a math problem like "Solve for x in equation 2x + 3 = 11.". 
            The model might first list the sub-problems as "Subtract 3 from both sides" and "Divide by 2". 
            It would then solve them in sequence to get the final answer.

        + `Self-refine prompting`
            + In this technique, the model is prompted to solve the problem, critique its solution, and then resolve the problem considering the problem, solution, and critique. 
            + The problem-solving process repeats until it reaches a predetermined reason to stop. *For example*, it could run out of tokens or time, or the model could output a stop token.

            + *For example*, 
            Imagine a user prompts a model, "Write a short essay on literature." 
            The model might draft an essay, critique it for lack of specific examples, and rewrite the essay to include specific examples. 
            This process would repeat until the essay is deemed satisfactory or a stop criterion is met.

        + `Directional-stimulus prompting`
            + This prompt engineering technique includes a hint or cue, such as desired keywords, to guide the language model toward the desired output.

            + *For example*, 
            If the prompt is to write a poem about love, the prompt engineer may craft prompts that include "heart," "passion," and "eternal." 
            The model might be prompted, "Write a poem about love that includes the words 'heart,' 'passion,' and 'eternal'." 
            This would guide the model to craft a poem with these keywords.

- [***Prompt Engineering best practices***](https://github.com/microsoft/generative-ai-for-beginners/tree/main/04-prompt-engineering-fundamentals#best-practices)
    + *Unambiguous prompts*
        + Clearly define the desired response in your prompt to avoid misinterpretation by the AI. 
        + *For instance*, 
        If you are asking for a novel summary, clearly state that you are looking for a summary, not a detailed analysis. 
        This helps the AI to focus only on your request and provide a response that aligns with your objective.

    + *Adequate context within the prompt*
        + Provide adequate context within the prompt and include output requirements in your prompt input, confining it to a specific format. 
        + *For instance*, 
        Say if we want a list of the most popular movies of the 1990s in a table. 
        To get the exact result, we should explicitly state how many movies we want to be listed and ask for table formatting.

    + *Balance between targeted information and desired output*
        + Balance simplicity and complexity in the prompt to avoid vague, unrelated, or unexpected answers. + A prompt that is too simple may lack context, while a prompt that is too complex may confuse the AI. 
        + This is especially important for complex topics or domain-specific language, which may be less familiar to the AI. 
        + Instead, use simple language and reduce the prompt size to make the question more understandable.

    + *Experiment and refine the prompt*
        + Prompt engineering is an iterative process. 
        + It's essential to experiment with different ideas and test the AI prompts to see the results. 
        + We may need multiple tries to optimize for accuracy and relevance. 
        + Continuous testing and iteration reduce the prompt size and help the model generate better output. 
        + There are no fixed rules for how the AI outputs information, so flexibility and adaptability are essential.

- [*PromptTempalets*](https://medium.com/datadreamers/langchain-genai-enabler-across-digitals-76f7381f029b)
    + A prompt template in LangChain is a class that can be used to generate prompts for LLMs. 
    + A prompt is a piece of text that is used to guide the LLM's response. In a High level Prompt templates will give an option to set the context of a LLM response. 
    + Prompt templates can be used to generate prompts for a variety of tasks, such as question answering, summarization, and creative writing.
    + *For Example,* <img src="https://miro.medium.com/v2/resize:fit:1100/format:webp/1*Pu8c0B2TKm0fMzPYLLjyXw.png" >

## [Chains](https://python.langchain.com/v0.1/docs/modules/chains/)
- In simple words, a chain is a sequence of calls, whether those calls are to LLMs, external tools, or data preprocessing steps.
- Chains helps to go beyond simple LLM interactions, they allow us to construct pipelines that combine different models and functionalities (External data, external API calls).
- By chaining components, we  can create powerful applications that can leverage the strengths of LLMs or integrate external data and services.
- The primary supported way to do this is with LangChain Expression Language (LCEL). LCEL is great for constructing your chains, but it's also nice to have chains used off-the-shelf. 
- There are two types of off-the-shelf chains that LangChain supports:
    + **LCEL chain constructors**
        + Chains that are built with LCEL. 
        + In this case, LangChain offers a higher-level constructor method. 
        + However, all that is being done under the hood is constructing a chain with LCEL.
        + *Commonly used Chains :*
            + `create_sql_query_chain` 
                + [For more details and examples Click Here](https://api.python.langchain.com/en/latest/chains/langchain.chains.sql_database.query.create_sql_query_chain.html#langchain.chains.sql_database.query.create_sql_query_chain)
                + ***When to use?*** 
                To construct a query for a SQL database from natural language.

                + ***Security Note***
                    + This chain generates SQL queries for the given database.
                    + To mitigate risk of leaking sensitive data, limit permissions to read and scope to the tables that are needed.

                + ***Arguments***
                    + `llm:` The language model to use. 
                    + `db:` The SQLDatabase to generate the query for. 
                    + `prompt:` The prompt to use. If none is provided, will choose one based on dialect. Defaults to None. See Prompt section below for more.
                    + `k:` The number of results per select statement to return. Defaults to 5.

                + ***Returns***
                    + A chain that takes in a question and generates a SQL query that answers that question.
    
    + [**Legacy Chains**](https://python.langchain.com/v0.1/docs/modules/chains/#legacy-chains)
        + Legacy Chains constructed by subclass from a legacy Chain class. 
        + These chains do not use LCEL under the hood but are the standalone classes.
        + *Commonly used Chains :*
            + `ConversationalRetrievalChain`
                + [For more details and examples Click Here](https://api.python.langchain.com/en/latest/chains/langchain.chains.conversational_retrieval.base.ConversationalRetrievalChain.html#langchain.chains.conversational_retrieval.base.ConversationalRetrievalChain)
                + This chain can be used to have conversations with a document. 
                + It takes in a *`question`* and *(optional) `previous conversation history`*. 
                + If there is a previous conversation history, it uses an LLM to rewrite the conversation into a query to send to a retriever (otherwise it just uses the newest user input). 
                + It then fetches those documents and passes them (along with the conversation) to an LLM to respond.
                + ***APPLICATIONS***
                    + Chat Applications with conversational history (chat history)
            
            + `RetrievalQA`
                + [For more details and examples Click Here](https://api.python.langchain.com/en/latest/chains/langchain.chains.retrieval_qa.base.RetrievalQA.html#langchain.chains.retrieval_qa.base.RetrievalQA)
                + This chain first does a retrieval step to fetch relevant documents. 
                + Then passes those documents into an LLM to generate a response.
                + ***APPLICATIONS***
                    + Chat Application
            
            + `RetrievalQAWithSourcesChain`
                + Does question answering over retrieved documents, and cites it sources. 
                + Use this when we want the *answer response to have sources in the text response*. 
                + Use this over `load_qa_with_sources_chain` when we want to use a retriever to fetch the relevant document as part of the chain (rather than pass them in).

            + `load_qa_with_sources_chain`
                + Does question answering over documents we pass in, and cites it sources. 
                + Use this when we want the *answer response to have sources in the text response*. 
                + Use this over `RetrievalQAWithSources` when we want to pass in the documents directly (rather than rely on a retriever to get them).

            + `StuffDocumentsChain`
                + [For more details and examples Click Here](https://api.python.langchain.com/en/latest/chains/langchain.chains.combine_documents.stuff.StuffDocumentsChain.html#langchain.chains.combine_documents.stuff.StuffDocumentsChain)
                + Chain that combines documents by stuffing into context.
                    + This chain takes a list of documents and first combines them into a single string. 
                    + It does this by formatting each document into a string with the `document_prompt` and then joining them together with `document_separator`. 
                    + It then adds that new string to the inputs with the variable name set by `document_variable_name`. Those inputs are then passed to the `llm_chain`.
                + ***APPLICATIONS***
                    + [Summarize Text](https://python.langchain.com/v0.2/docs/tutorials/summarization/#stuff)
            
            + `ReduceDocumentsChain`
                + This chain combines documents by iterative reducing them. 
                + It groups documents into chunks (less than some context length) and then passes them into an LLM. 
                + It then takes the responses and continues to do this until it can fit everything into one final LLM call. 
                + It is useful when you have a lot of documents, you want to have the LLM run over all of them, and you can do it in parallel.
                + ***APPLICATIONS***
                    + [Summarize Text](https://python.langchain.com/v0.2/docs/tutorials/summarization/#map-reduce)


            + `MapReduceDocumentsChain`
                + This chain first passes each document through an LLM, then reduces them using the `ReduceDocumentsChain`. It is useful in the same situations as ReduceDocumentsChain, but does an initial LLM call before trying to reduce the documents.
                + ***APPLICATIONS***
                    + [Summarize Text](https://python.langchain.com/v0.2/docs/tutorials/summarization/#map-reduce)
            
            + `RefineDocumentsChain`
                + This chain collapses documents by generating an initial answer based on the first document and then looping over the remaining documents to refine its answer. This operates sequentially, so it cannot be parallelized. It is useful in similar situations as `MapReduceDocuments` Chain, but for cases where you want to build up an answer by refining the previous answer (rather than parallelizing calls).
                + ***APPLICATIONS***
                    + [Summarize Text](https://python.langchain.com/v0.2/docs/tutorials/summarization/#refine)

    + There are many chains as per requirements, to look for more- [click here](https://python.langchain.com/v0.1/docs/modules/chains/)          

### [How to fix if the document is super long that it **exceeds the token limit**?]((https://towardsdatascience.com/4-ways-of-question-answering-in-langchain-188c6707cc5a))
- There are two ways to fix it:
    + *Solution 1:* ***Chain Type***
        + The default ***chain_type = "stuff"*** uses ALL of the text from the documents in the prompt. 
        + It actually doesn't work when retrieved document is huge because it exceeds the token limit and causes rate-limiting errors. 
        + i.e., why we had to use other chain types for example "map_reduce". What are the other chain types?
            + ***map_reduce***
                + It separates texts into batches.
                + As an example, we can define batch size in *`llm = OpenAI(batch_size=5)`*.
                + It then feeds each batch with the question to LLM separately, and comes up with the final answer based on the answers from each batch.
                + **Parallel in nature**.

            + ***refine***
                + It separates texts into batches.
                + It then feeds the first batch to LLM. *(This generates answer-1)*
                + Then feeds the answer *(answer-1)* and the second batch to LLM. 
                + It refines the answer by going through all the batches.
                + **Sequential in nature**.

            + ***map-rerank***
                + It separates texts into batches.
                + It then feeds each batch to LLM, returns a score of how fully it answers the question, and comes up with the final answer based on the high-scored answers from each batch.
                + **Parallel in nature**, but depends on *score*.

    + *Solution 2:* ***Retrieval Chains***
        + One issue with using ALL of the text is that it can be very costly because you are feeding all the texts to OpenAI API and the API is charged by the number of tokens. 
        + A better solution is to retrieve relevant text chunks first and only use the relevant text chunks in the language model. 
        + Available Retrieval chains are:
            + `ConversationalRetrievalChain` 
            **ConversationalRetrievalChain** = *conversation memory* + *RetrievalQAChain*
            + `RetrievalQA`
            + `RetrievalQAWithSourcesChain`
            + `load_qa_with_sources_chain`

- [StackOverFlow RetrievalQA Example](https://stackoverflow.com/questions/77521745/map-reduce-prompt-with-retrievalqa-chain) or [web_page_summerizer](https://github.com/amitoshacharya/Generative-AI-Project/blob/main/GenAI-%20Learning/web_tool_summerizer%20copy.ipynb)
    ```python
    qa_template = """
    Use the following information from the context (separated with <ctx></ctx>) to answer the question.
    Answer in German only, because the user does not understand English! \
    If you don't know the answer, answer with "Unfortunately, I don't have the information." \
    If you don't find enough information below, also answer with "Unfortunately, I don't have the information." \
    ------
    <ctx>
    {context}
    </ctx>
    ------
    <hs>
    {chat_history}
    </hs>
    ------
    {question}
    Answer:
    """

    prompt = PromptTemplate(template=qa_template,
                                input_variables=['context','history', 'question'])
    combine_custom_prompt='''
    Generate a summary of the following text that includes the following elements:

    * A title that accurately reflects the content of the text.
    * An introduction paragraph that provides an overview of the topic.
    * Bullet points that list the key points of the text.
    * A conclusion paragraph that summarizes the main points of the text.

    Text:`{context}`
    '''


    combine_prompt_template = PromptTemplate(
        template=combine_custom_prompt, 
        input_variables=['context']
    )

    chain_type_kwargs={
            "verbose": True,
            "question_prompt": prompt,
            "combine_prompt": combine_prompt_template,
            "combine_document_variable_name": "context",
            "memory": ConversationSummaryMemory(
                llm=OpenAI(),
                memory_key="history",
                input_key="question",
                return_messages=True)}


    refine = RetrievalQA.from_chain_type(llm=OpenAI(),
                                        chain_type="map_reduce",
                                        return_source_documents=True,
                                        chain_type_kwargs=chain_type_kwargs,
                                        retriever=big_chunks_retriever,
                                        verbose=True)
    ```

## [Agents](https://python.langchain.com/v0.1/docs/modules/agents/)
- [YouTube video](https://youtu.be/cQUUkZnyoD0?t=2717)
- [***What are Agents?***]((https://towardsdatascience.com/building-a-simple-agent-with-tools-and-toolkits-in-langchain-77e0f9bd1fa5))
    + The core idea of agents is to use a language model to choose a ***sequence of actions to take***. 
    + ***Sequence of Action***
        + Here are few sample sequence of actions that can be taken (in case of agents) or coded (in case of chains):
            + *Text generation* 
            The agent generates text based on a prompt or input.
            + *Text processing* 
            The agent processes the generated text, such as tokenizing, part-of-speech tagging, or named entity recognition.
            + *Knowledge retrieval* 
            The agent retrieves relevant knowledge or information from a database or knowledge graph.
            + *Reasoning* 
            The agent applies logical or inferential reasoning to the retrieved knowledge.
            + *Decision-making* 
            The agent makes a decision based on the reasoning step.
            + *Action execution* 
            The agent executes an action, such as sending a message, making a recommendation, or triggering another agent.
            + *Feedback processing* 
            The agent processes feedback from the environment or user.

    + In chains, a sequence of actions is hardcoded (in code). In agents, a language model (LLM) is used as a reasoning engine to determine which actions to take and in which order.

    + ***Components of Agents***
        + From the definition of AI Agents, we would get something along the lines of *"An entity that is able to perceive its environment, act on its environment, and make intelligent decisions about how to reach a goal it has been given, as well as the ability to learn as it goes"*
        + Thus there are 3 components of Agents:
            + ***`Large Language Model (LLM)`***
                + `The brains of a LangChain agent are an LLM.`
                + It is the LLM that is used to ***reason*** about the best way to carry out the ask requested by a user.

            + [***`Tools`***](https://python.langchain.com/v0.1/docs/modules/tools/)
                + In order to carry out its task, and operate on things and retrieve information, the agent has what are called Tool's in LangChain.
                + It is through these tools that it is able to interact with its ***environment***.
                + The tools are basically just methods/classes the agent has access to. 
                + With these tools, agent can do things like;
                    + interact with a Stock Market index over an API
                    + update a Google Calendar event
                    + run a query against a database
                    + etc.... 
                + We can build out tools as needed, depending on the nature of tasks we are trying to carry out with the agent to fulfil.
                    + Here are the list of [Built-in Tools](https://python.langchain.com/v0.1/docs/integrations/tools/).
                    + Although built-in tools are useful, it's highly likely that we'll have to define our own tools i.e; [Custom Tools](https://python.langchain.com/v0.1/docs/modules/tools/custom_tools/).

            + ***`Toolkit`***
                + A collection of Tools in LangChain are called a Toolkit. 
                + Implementation wise, this is literally just an array of the Tools that are available for the agent.
    
    +  ***NOTE:*** 
    *LangChain's [library of Toolkits](https://python.langchain.com/docs/integrations/toolkits/) (like: AINetwork, Azure AI Services, CSV, Gmail, JSON, Jira, Python, SQL Database, Spark SQL etc.)  for agents to use, listed on their Integrations page, are sets of Tools built by the community for people to use, which could be an early example of agent type libraries built by the community.*

    + As such, the high level overview of an agent in LangChain looks something like this: <img src="https://miro.medium.com/v2/resize:fit:828/format:webp/1*Hah4yA9A0rniFn2tEDyNrA.png" width=400 height=300>

    + So, at a basic level, an agent needs
        + `LLM`, to act as its brain, and to give it its reasoning abilities.
        + `Tools`, so that it can interact with the environment around it and achieve its goals.

    + [*Building A Simple Agent*](https://towardsdatascience.com/building-a-simple-agent-with-tools-and-toolkits-in-langchain-77e0f9bd1fa5)

- [***What are Agent Types?***](https://python.langchain.com/v0.1/docs/modules/agents/agent_types/)
    + LangChain has a number of different agents types that can be created, with different reasoning powers and abilities.
    + LangChain categorizes agents based on several dimensions:
        + **Intended Model Type**
            + Whether this agent is intended for Chat models or LLMs:
                + `Chat` Models (takes in messages, outputs message)
                + `LLMs` (takes in string, outputs string). 
            + The main thing this affects is the prompting strategy used. 
            + **Note:** We can use an agent with a different type of model than it is intended for, but it likely won't produce results of the same quality.

        + **Supports Chat History**
            + Whether or not these agent types support chat history. 
            + If it does, that means it can be used as a chatbot. 
            + If it does not, then that means it's more suited for single tasks. 
            + Supporting chat history generally requires better models, so earlier agent types aimed at worse models may not support it.

        + **Supports Multi-Input Tools**
            + Whether or not these agent types support tools with multiple inputs. 
            + If a tool only requires a single input, it is generally easier for an LLM to know how to invoke it. 
            + Therefore, several earlier agent types aimed at worse models may not support them.

        + **Supports Parallel Function Calling**
            + Having an LLM call multiple tools at the same time can greatly speed up agents whether there are tasks that are assisted by doing so. 
            + However, it is much more challenging for LLMs to do this, so some agent types do not support this.

        + **Required Model Params**
            + Whether this agent requires the model to support any additional parameters. 
            + Some agent types take advantage of things like OpenAI function calling, which require other model parameters. 
            + If none are required, then that means that everything is done via prompting.

    + [***Agent Types and When to Use***](https://python.langchain.com/v0.1/docs/modules/agents/agent_types/)
        + [*OpenAI functions*](https://python.langchain.com/v0.1/docs/modules/agents/agent_types/openai_functions_agent/) (***Legacy***)
            + `Intended Model Type:` *Chat*
            + `Supports Chat History:` *Yes*
            + `Supports Multi-Input Tools:` *Yes*
            + `Required Model Params:` *functions*	
            + `When to Use:`
                + *If you are using an OpenAI model, or an open-source model that has been finetuned for function calling and exposes the same functions parameters as OpenAI.*
                + *Generic Tool Calling agent recommended instead*
            + `API:` 
            [*langchain.agents.openai_functions_agent.base.create_openai_functions_agent*](https://api.python.langchain.com/en/latest/agents/langchain.agents.openai_functions_agent.base.create_openai_functions_agent.html)

        + [*Tool Calling*](https://python.langchain.com/v0.1/docs/modules/agents/agent_types/tool_calling/)
            + `Intended Model Type:` *Chat*
            + `Supports Chat History:` *Yes*
            + `Supports Multi-Input Tools:` *Yes*
            + `Supports Parallel Function Calling:` *Yes*
            + `Required Model Params:` *tools*
            + `When to Use:`
                + *If we are using a tool-calling model*
            + `API:`
            [*langchain.agents.tool_calling_agent.base.create_tool_calling_agent*](https://api.python.langchain.com/en/latest/agents/langchain.agents.tool_calling_agent.base.create_tool_calling_agent.html)



        + [*OpenAI tools*](https://python.langchain.com/v0.1/docs/modules/agents/agent_types/openai_tools/) (***Legacy***)
            + `Intended Model Type:` *Chat*
            + `Supports Chat History:` *Yes*
            + `Supports Multi-Input Tools:` *Yes*
            + `Supports Parallel Function Calling:` *Yes*
            + `Required Model Params:` *tools*
            + `When to Use:`
                + *If we are using a recent OpenAI model (1106 onwards).* 
                + *Generic Tool Calling agent recommended instead.*
            + `API:`
            [*langchain.agents.openai_tools.base.create_openai_tools_agent*](https://api.python.langchain.com/en/latest/agents/langchain.agents.openai_tools.base.create_openai_tools_agent.html)

        + [*XML Agent*](https://python.langchain.com/v0.1/docs/modules/agents/agent_types/xml_agent/)
            + `Intended Model Type:` *LLM*
            + `Supports Chat History:` *Yes*	
            + `When to Use:`
                + *If we are using Anthropic models, or other models good at XML.*
            + `API:`
            [*langchain.agents.xml.base.create_xml_agent*](https://api.python.langchain.com/en/latest/agents/langchain.agents.xml.base.create_xml_agent.html)

        + [*JSON Chat Agent*](https://python.langchain.com/v0.1/docs/modules/agents/agent_types/json_agent/)
            + `Intended Model Type:` *Chat*
            + `Supports Chat History:` *Yes*
            + `When to Use:`
                + *If you are using a model good at JSON.*
            + `API:`
            [*langchain.agents.json_chat.base.create_json_chat_agent*](https://api.python.langchain.com/en/latest/agents/langchain.agents.json_chat.base.create_json_chat_agent.html)

        + [*Structured chat*](https://python.langchain.com/v0.1/docs/modules/agents/agent_types/structured_chat/)
            + `Intended Model Type:` *Chat*
            + `Supports Chat History:` *Yes*
            + `Supports Multi-Input Tools:` *Yes*
            + `When to Use:`
                + *If we need to support tools with multiple inputs*
            + `API:`
            [*langchain.agents.structured_chat.base.create_structured_chat_agent*](https://api.python.langchain.com/en/latest/agents/langchain.agents.structured_chat.base.create_structured_chat_agent.html)

        + [*ReAct agent*](https://python.langchain.com/v0.1/docs/modules/agents/agent_types/react/)
            + `Intended Model Type:` *LLM*
            + `Supports Chat History:` *Yes*	
            + `When to Use:`
                + *If we are using a simple model.*
            + `API:`
            [*langchain.agents.react.agent.create_react_agent*](https://api.python.langchain.com/en/latest/agents/langchain.agents.react.agent.create_react_agent.html)

        + [*Self-ask with search*](https://python.langchain.com/v0.1/docs/modules/agents/agent_types/self_ask_with_search/)
            + `Intended Model Type:` *LLM*	
            + `When to Use:`
                + *If we are using a simple model and only have one search tool.*
            + `API:`
            [*langchain.agents.self_ask_with_search.base.create_self_ask_with_search_agent*](https://api.python.langchain.com/en/latest/agents/langchain.agents.self_ask_with_search.base.create_self_ask_with_search_agent.html)

- ***AgentExecutor***
    + In order to run agents in LangChain, we **cannot** just call a "run" type method on them directly. They need to be run via an AgentExecutor.
    + The agent executor is the runtime for an agent. 
    + This is what actually calls the agent, executes the actions it chooses, passes the action outputs back to the agent, and repeats. 
    + In pseudocode, this looks roughly like:
        ```
        next_action = agent.get_action(...)
        while next_action != AgentFinish:
            observation = run(next_action)
            next_action = agent.get_action(..., next_action, observation)
        return next_action
        ```

    + So they are basically a while loop that keep's calling the next action methods on the agent, until the agent has returned its final response.
    + To setup an agent inside the agent executor. We pass to following parameters:
        + `Agent`
        + `Tools` or Toolkit
        + `Verbose`: Set to *True* so we can get an idea of what the agent is doing as it is processing our request.
            ```
            agent_executor = AgentExecutor(agent=agent, tools=toolkit, verbose=True)
            ```
    + To run the agent we have to invoke Agent Executor
        ```
        result = agent_executor.invoke({"input": "what is 1 + 1"})
        ```

## [ReAct Agent](https://python.langchain.com/v0.1/docs/modules/agents/agent_types/react/)
- [Overview](https://react-lm.github.io/)
- ***ReAct***- *Synergizing Reasoning and Acting in Language Models* <img src=https://react-lm.github.io/files/diagram.png>

- We can use ReAct agent, 
    + if we are using a simple language model
    + and also, if we are looking for agent that supports chat history.
- **Steps to implement ReAct Agent involves:** *Note the [Use Case considered below is using GenAI over SQl Database.](https://github.com/amitoshacharya/Generative-AI-Project/blob/main/GenAI-%20Learning/sql_chain_%26_agent_explore.ipynb)*

    + **`Defining Large Language Model (LLM)`**
        + Here, defining AzureChatOpenAI

        ```python
        from langchain_openai import AzureChatOpenAI

        llm = AzureChatOpenAI(model = OPENAI_MODEL,
                              azure_deployment = OPENAI_DEPLOYMENT,
                              azure_endpoint = OPENAI_ENDPOINT,
                              api_version = OPENAI_API_VERSION,
                              api_key = OPENAI_API_KEY,
                              temperature = 0
                            )
        ```

    + **`Difining Data Connection`**
        + [***Dialect***](https://docs.sqlalchemy.org/en/13/dialects/index.html)
            + The **dialect** is the system SQLAlchemy uses to communicate with various types of DBAPI implementations and databases.
            + *Included Dialects* &#8594; `PostgreSQL`, `MySQL`, `SQLite`, `Oracle`, `Microsoft SQL Server`
            + *External Dialects* &#8594; `Amazon Redshift (via psycopg2)`, `Apache Drill`, `Apache Druid`, `Apache Hive and Presto`, `Apache Solr`, `Google BigQuery`, `SAP Hana`, `Snowflake`, .. etc.

        + ***Table Info***
            + Information about all tables in the database.

            ```python
            from langchain_community.utilities import SQLDatabase

            ## data_base_uri this is the uri of the SQL db. And is combination of 'dialect' and 'db_path'
            db = SQLDatabase.from_uri(database_uri = f"sqlite:///{data_base_path}")
            ```

    + **`Initialize Toolkit or list of Tools`**

        ```python
        from langchain_community.agent_toolkits import SQLDatabaseToolkit

        toolkit = SQLDatabaseToolkit(llm = llm, db = db)
        sql_tools = toolkit.get_tools()
        ```

    + **`Prompt Engineering`**
        + We can create custom prompt.

            ```python
            from langchain_core.prompts import PromptTemplate

            template = """
            You are an AI agent, designed to interact with the SQL database. You are an agent that does a reasoning step before the acting. Given an input question, create a syntactically correct dialect query to run, then look at the results of the query and return the answer. Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results. You can order the results by a relevant column to return the most interesting examples in the database. Never query for all the columns from a specific table, only ask for the relevant columns given the question. You have access to tools for interacting with the database. Only use the below tools. Only use the information returned by the below tools to construct your final answer. You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again. DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database. If the question does not seem related to the database, just return "I am Sorry, I only answer questions related to the database" as the answer. If you come across the Final Answer immediately stop Thought, Action Process and return the answer framing a very good sentence. 
            
            Answer the following questions as best you can. You have access to the following tools: {tools}  

            Use the following format:  
            Question: the input question you must answer 
            Thought: you should always think about what to do 
            Action: the action to take, should be one of [{tool_names}] 
            Action Input: the input to the action 
            Observation: the result of the action ... (this Thought/Action/Action Input/Observation can repeat N times) 
            Final Thought: I now know the final answer 
            Final Answer: the final answer to the original input question  

            Begin!  

            Question: {input} 
            Thought:{agent_scratchpad}
            """

            prompt = PromptTemplate(template = template, input_variables = ['agent_scratchpad', 'input', 'tool_names', 'tools'])
            ```
        
        + We can also pull existing prompts from [Langchain Hub](https://docs.smith.langchain.com/how_to_guides/prompts/langchain_hub).
            + Hub is a collection of pre-existing prompt.
            + Here you'll find all of the publicly listed prompts in the LangChain Hub. You can search for prompts by name, handle, use cases, descriptions, or models.
            + Langchain Hub &#8594; [Click here](https://smith.langchain.com/hub)

            ```python
            from langachain import hub

            prompt = hub.pull("sharsha315/custom_react_sql_agent")
            ```

    + **`Create Agent`**
        + Here, we are creating a ReAct Agent

            ```python
            from langchain.agents import create_react_agent

            agent = create_react_agent(llm =llm, tools = sql_tools, prompt = prompt)
            ```

    + **`Run Agent`** &#8594; *AgentExecutor*
        + The agent executor is the runtime for an agent. 
        + This is what actually calls the agent, executes the actions it chooses, passes the action outputs back to the agent, and repeats.
        + Set ***verbose*** to *True*, so we can get an idea of what the agent is doing as it is processing our request.
        + **Important Note:** 
            + When interacting with agents it is really important to set the ***max_iterations*** parameters because agents can get stuck in infinite loops that consume plenty of tokens. 
            + The <ins>default value is **15**</ins> to allow for many tools and complex reasoning but for most applications you should keep it much lower.

                ```python
                from langchain.agents import AgentExecutor

                agent_executor = AgentExecutor(agent=agent, 
                                            tools= sql_tools,
                                            verbose=True, 
                                            handle_parsing_errors=True,
                                            max_iterations=15)

                question = f"What are the records under 'Home Appliances' category of table name - '{table_name}' got sold in Dec 2019." +\
                "Return following columns only: id, date of order, category of product, product, time they order and there address." +\
                "Return the results in Table format."
                
                result = agent_executor.invoke({"input":question})
                ```
        
        + ***Using with chat history***

            ```python
            result = agent_executor.invoke({
                                                "input": "what's my name? Only use a tool if needed, otherwise respond with Final Answer",
                                                # Notice that chat_history is a string, since this prompt is aimed at LLMs, not chat models
                                                "chat_history": "Human: Hi! My name is Bob\nAI: Hello Bob! Nice to meet you",
                                            }
                                        )
            ```
        
- [**Dealing with high-cardinality columns**](https://python.langchain.com/v0.2/docs/tutorials/sql_qa/#dealing-with-high-cardinality-columns)

    
        


