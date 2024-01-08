import gradio as gr
import time
from transformers import pipeline

model_name = "youlun77/finetuning-sentiment-model-25000-samples"
classifier_distilbert = pipeline('sentiment-analysis', model=model_name)

def query_distilbert(payload):
    start = time.time()
    response = classifier_distilbert(payload["inputs"])
    return response[0], (time.time()-start)

def demo_func_distil_label(input_text):
    sentences = input_text.split('.')
    result =[]
    current_sentence = ''
    total_score = 0
    total_time = 0

    for sentence in sentences:
        if sentence.strip() and len(current_sentence + sentence) <= 512:
            current_sentence += sentence
        else:
            result.append(current_sentence.strip())
            current_sentence = sentence
    if current_sentence:
        result.append(current_sentence.strip())

    result = [sentence for sentence in result if sentence]
    for part in result:
        output, duration = query_distilbert({
	    "inputs": str(part),
        })
        print(f'D:{output}')
        total_time += duration
        if "label" in output and output["label"] == "LABEL_0":
            total_score += output["score"]

    avg=total_score/len(result)
    if avg<=0.5:
        return "POSTIVE!", total_time
    else:
        return "NEGATIVE:(", total_time
    
model_name = "youlun77/finetuning-sentiment-model-25000-samples-BERT"
classifier = pipeline('sentiment-analysis', model=model_name)

def query_bert(payload):
    start = time.time()
    response = classifier(payload["inputs"])
    return response[0], time.time()-start
	
def demo_func_bert_label(input_text):
    sentences = input_text.split('.')
    result =[]
    current_sentence = ''
    total_score = 0.0
    total_time = 0

    for sentence in sentences:
        if sentence.strip() and len(current_sentence + sentence) <= 512:
            current_sentence += sentence
        else:
            result.append(current_sentence.strip())
            current_sentence = sentence
    if current_sentence:
        result.append(current_sentence.strip())

    for part in result:
        output, duration = query_bert({
	    "inputs": str(part),
        })
        print(f'{output}')
        total_time += duration
        if "label" in output and output["label"] == "LABEL_0":
            total_score += output["score"]

    avg=total_score/len(result)
    print(f'avg = {avg}')
    if avg<=0.5:
        return "POSTIVE!", total_time
    else:
        return "NEGATIVE:(", total_time
    
def combine(input):
    a, b = demo_func_distil_label(input)
    c, d = demo_func_bert_label(input)
    return a,b,c,d


with gr.Blocks() as distilbot:
    input_text = gr.Textbox()
    with gr.Row():
        with gr.Column(scale=1):
            distil_output_label = gr.Label(label = "distilbert output")
            distil_time_text = gr.Textbox(label = "distilbert generate time")
        with gr.Column(scale=1):
            bert_output_label = gr.Label(label = "bert output")
            bert_time_text = gr.Textbox(label = "bert generate time")

    input_text.submit(demo_func_bert_label, 
                      inputs = input_text, 
                      outputs = [bert_output_label, bert_time_text]) \
                .then(
                        demo_func_distil_label, 
                        inputs = input_text, 
                        outputs = [distil_output_label, distil_time_text]
                      )
    
    gr.Markdown("#### Long Text Examples")
    gr.Examples(
        examples = ["""
Introduction:

In the vast landscape of cinema, certain films emerge as timeless masterpieces, captivating audiences with their compelling narratives, impeccable craftsmanship, and unforgettable performances. One such cinematic gem that has left an indelible mark on the hearts of viewers is "The Whispering Shadows." Directed by the visionary filmmaker, Elena Rodriguez, this movie stands as a testament to the power of storytelling and the magic of the silver screen.

The Captivating Narrative:

At the heart of "The Whispering Shadows" lies a narrative that transcends the boundaries of conventional storytelling. Rodriguez weaves a tapestry of emotions, mystery, and intrigue that keeps the audience on the edge of their seats from the opening scene to the closing credits. The screenplay, penned with meticulous attention to detail, unfolds like a well-crafted novel, drawing the viewers into a world where reality and fantasy seamlessly coexist.

The film follows the journey of the protagonist, played with unparalleled brilliance by the seasoned actor, Julia Turner. Turner's performance is nothing short of extraordinary, as she effortlessly navigates the complexities of her character, bringing depth and authenticity to every scene. Her portrayal of the lead character is a tour de force, earning her accolades and leaving an indelible imprint on the audience's collective consciousness.

Visual Extravaganza:

"The Whispering Shadows" is a visual feast for cinephiles and art enthusiasts alike. Cinematographer Carlos Hernandez employs a masterful use of light and shadow, creating a visual symphony that enhances the emotional resonance of each frame. The film's exquisite cinematography not only serves the narrative but also elevates it to a level of visual poetry seldom witnessed in contemporary cinema.

The set design and art direction are equally commendable, transporting the audience to a mesmerizing world where every detail is meticulously crafted. From the enchanting landscapes to the intricately designed interiors, the film's visual elements contribute to the overall immersive experience, making it a cinematic spectacle that lingers in the viewer's memory long after the credits roll.

A Musical Odyssey:

The ethereal soundtrack of "The Whispering Shadows" is a symphony of emotions, skillfully composed by the maestro, Antonio Ramirez. The music becomes an integral part of the storytelling, evoking a myriad of emotions that resonate with the audience on a profound level. Ramirez's score complements the narrative, creating a sonic landscape that enhances the film's emotional impact and underscores its thematic depth.

Themes and Symbolism:

Beneath the surface of its gripping storyline and stunning visuals, "The Whispering Shadows" explores profound themes that resonate with the human experience. Rodriguez skillfully incorporates symbolism and allegory, inviting the audience to ponder on the deeper meanings hidden within the narrative. The film becomes a canvas on which universal truths and existential questions are painted, encouraging introspection and contemplation.

Social Relevance and Timeliness:

"The Whispering Shadows" is not merely a cinematic escapade; it is a mirror reflecting the zeitgeist of its time. Rodriguez deftly addresses social issues, seamlessly integrating them into the narrative without compromising the film's entertainment value. The movie becomes a vehicle for social commentary, sparking conversations and raising awareness about pressing societal concerns.

Conclusion:

In conclusion, "The Whispering Shadows" stands as a cinematic triumph, a symphony of sight and sound that transcends the boundaries of traditional filmmaking. Elena Rodriguez's visionary direction, coupled with stellar performances, exquisite visuals, and a captivating narrative, elevates the movie to the pantheon of timeless classics. This cinematic masterpiece is not just a film; it is an experience that resonates with the audience, leaving an enduring legacy in the annals of cinematic history. "The Whispering Shadows" is a testament to the transformative power of storytelling and the enchanting magic of cinema.
                    """],
        inputs = input_text,
        outputs = [distil_output_label, distil_time_text,bert_output_label, bert_time_text],
        fn = combine,
        cache_examples=False,
        run_on_click=True
    )
    gr.Markdown("#### Long Text Examples")
    gr.Examples(
        examples = ["""
Introduction:

While the realm of cinema often produces masterpieces that leave an indelible mark on audiences, there are instances when a film falls short of its potential, leaving viewers with a sense of disappointment and unfulfilled expectations. "The Whispering Shadows," directed by Elena Rodriguez, unfortunately, falls into this category. Despite its initial promise, the film is marred by various shortcomings, ranging from a convoluted plot to lackluster performances, ultimately preventing it from achieving the cinematic greatness it aspired to.

The Confused Narrative:

At the core of "The Whispering Shadows" lies a narrative that, rather than captivating the audience, tends to confuse and alienate them. Rodriguez's attempt to blend reality with fantasy results in a convoluted and disjointed storyline. The film's plot unfolds in a non-linear fashion, making it challenging for viewers to connect the dots and fully engage with the characters and their motivations. The narrative lacks a clear and cohesive structure, leaving audiences perplexed rather than enthralled.

The protagonist, portrayed by Julia Turner, is given a character arc that lacks depth and fails to resonate emotionally. The screenplay's reliance on clichés and predictable twists further diminishes the impact of what could have been a compelling and original story. As a result, the film's narrative becomes a maze of unresolved subplots, leaving viewers grappling with a sense of dissatisfaction and frustration.

Underwhelming Performances:

While the cast of "The Whispering Shadows" boasts talent, the performances fail to live up to the expectations set by the film's premise. Julia Turner, despite her reputation as a seasoned actor, delivers a lackluster performance that lacks the emotional depth required for the role. The character's journey is overshadowed by Turner's inability to convey the complexity and nuance necessary to make the audience empathize with her plight.

The supporting cast, though competent, is hindered by a script that offers limited opportunities for character development. The lack of chemistry among the ensemble cast further diminishes the film's impact, making it challenging for viewers to invest emotionally in the relationships portrayed on screen. The performances, rather than elevating the narrative, become a hindrance to the film's overall success.

Visual Dissonance:

While some may praise the cinematography of "The Whispering Shadows," it is not immune to criticism. The film's visual elements, though aesthetically pleasing, often feel disconnected from the narrative, creating a sense of dissonance. Carlos Hernandez's use of light and shadow, while impressive in isolation, fails to complement the storytelling, at times overshadowing the characters and their interactions.

The set design, although intricate, borders on excessive, distracting viewers from the central narrative. The elaborate visuals, while visually stimulating, do not always serve the story, giving the impression of a film more concerned with style than substance. The result is a visual landscape that, rather than enhancing the viewer's experience, detracts from the overall cohesiveness of the film.

Musical Misses:

While a film's soundtrack can enhance its emotional resonance, Antonio Ramirez's score for "The Whispering Shadows" fails to leave a lasting impression. The music, though competent, lacks the originality and innovation needed to elevate the film to a higher cinematic plane. The score, at times, feels formulaic and uninspired, failing to capture the intricacies of the narrative or evoke a strong emotional response from the audience.

Themes Lost in Translation:

Despite its attempts to incorporate symbolism and allegory, "The Whispering Shadows" struggles to convey its intended themes effectively. The film's exploration of profound subjects often feels forced and contrived, leaving audiences with a sense of ambiguity rather than enlightenment. The thematic elements, instead of adding depth to the narrative, become a source of confusion, further contributing to the film's overall lack of clarity.

Social Commentary Lost in Translation:

While the film aims to address social issues, the execution of this endeavor falls flat. The integration of social commentary feels forced and superficial, lacking the nuance and depth required to make a meaningful impact. Rather than sparking meaningful conversations, the film's attempts at addressing societal concerns come across as token gestures, leaving viewers with a sense of missed opportunities.

Conclusion:

In conclusion, "The Whispering Shadows" fails to live up to the expectations set by its premise and talented cast. The convoluted narrative, underwhelming performances, visual dissonance, and lackluster musical score collectively contribute to the film's inability to achieve greatness. While it is not without its merits, the flaws of "The Whispering Shadows" outweigh its strengths, resulting in a cinematic experience that falls short of its potential. As audiences continue to seek films that resonate on a profound level, this particular endeavor by Elena Rodriguez serves as a cautionary tale about the importance of cohesive storytelling, compelling performances, and a harmonious marriage of visual and narrative elements in the world of cinema.
                    """],
        inputs = input_text,
        outputs = [distil_output_label, distil_time_text,bert_output_label, bert_time_text],
        fn = combine,
        cache_examples=False,
        run_on_click=True
    )
    gr.Markdown("#### Text Examples")
    gr.Examples(
        examples = ["I’ve missed more than 9000 shots in my career. I’ve lost almost 300 games. 26 times, I’ve been trusted to take the game winning shot and missed. I’ve failed over and over and over again in my life. And that is why I succeed."],
        inputs = input_text,
        outputs = [distil_output_label, distil_time_text,bert_output_label, bert_time_text],
        fn = combine,
        cache_examples=False,
        run_on_click=True
    )   
    gr.Markdown("#### Text Examples")
    gr.Examples(
        examples = ["You have to put in a lot of sacrifice and effort for sometimes little reward but you have to know that, if you put in the right effort, the reward will come."],
        inputs = input_text,
        outputs  = [distil_output_label, distil_time_text,bert_output_label, bert_time_text],
        fn = combine,
        cache_examples=False,
        run_on_click=True
    )   
    gr.Markdown("#### Text Examples")
    gr.Examples(
        examples = ["I am doomed to remember a boy with a wrecked voice—not because of his voice, or because he was the smallest person I ever knew, or even because he was the instrument of my mother's death, but because he is the reason I believe in God; I am a Christian because of Owen Meany"],
        inputs = input_text,
        outputs = [distil_output_label, distil_time_text,bert_output_label, bert_time_text],
        fn = combine,
        cache_examples=False,
        run_on_click=True
    )   
      
if __name__ == "__main__":
    distilbot.launch()