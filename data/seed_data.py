"""Seed data module for ClinIQ medical RAG system.

Contains a dataset of 25 medically accurate Q&A pairs covering five common health topics.
Sourced in spirit from consumer health references like MedlinePlus.
Maintains a factual, cautious tone with appropriate medical disclaimers.
"""

from typing import Dict, List

MEDICAL_QA: List[Dict[str, str]] = [
    # =========================================================================
    # Topic 1: Diabetes
    # =========================================================================
    {
        "topic": "Diabetes",
        "question": "What are the early warning signs of type 2 diabetes?",
        "answer": (
            "Early signs of type 2 diabetes may include increased thirst, frequent urination, "
            "unexplained fatigue, and blurred vision. Some individuals may also experience "
            "slow-healing cuts or frequent skin infections. These symptoms often develop gradually "
            "over months and might go unnoticed initially. If you notice persistent signs of high "
            "blood sugar, consulting a healthcare provider for a blood glucose evaluation is recommended."
        ),
    },
    {
        "topic": "Diabetes",
        "question": "What is the difference between Type 1 and Type 2 diabetes?",
        "answer": (
            "Type 1 diabetes is an autoimmune condition where the body's immune system mistakenly "
            "attacks insulin-producing cells in the pancreas. Type 2 diabetes occurs when the body "
            "becomes resistant to insulin or fails to produce sufficient quantities of it. While Type 1 "
            "diabetes typically develops during childhood or adolescence, Type 2 is more commonly diagnosed "
            "in adulthood. Both conditions require ongoing clinical management and monitoring to prevent "
            "long-term health complications."
        ),
    },
    {
        "topic": "Diabetes",
        "question": "How does dietary management help in controlling blood sugar levels?",
        "answer": (
            "Managing dietary carbohydrate intake helps maintain steady blood glucose levels throughout "
            "the day. Consuming foods rich in dietary fiber, such as vegetables, whole grains, and legumes, "
            "slows carbohydrate absorption and minimizes sudden blood sugar spikes. Pairing complex "
            "carbohydrates with lean proteins and healthy fats further supports balanced metabolism. "
            "Individuals managing diabetes should work with a registered dietitian or physician to "
            "establish an individualized nutrition plan."
        ),
    },
    {
        "topic": "Diabetes",
        "question": "What is gestational diabetes and how is it monitored?",
        "answer": (
            "Gestational diabetes is a form of elevated blood sugar that develops during pregnancy in "
            "individuals without a prior history of diabetes. It is routinely screened for using blood "
            "glucose testing between the 24th and 28th weeks of pregnancy. Management typically involves "
            "dietary adjustments, blood glucose self-monitoring, and moderate physical activity as approved "
            "by a doctor. In most cases, blood sugar levels return to normal after delivery, though long-term "
            "monitoring remains beneficial."
        ),
    },
    {
        "topic": "Diabetes",
        "question": "What steps should be taken if someone experiences symptoms of hypoglycemia?",
        "answer": (
            "Hypoglycemia, or low blood sugar, may cause symptoms such as shakiness, sweating, rapid "
            "heartbeat, dizziness, and sudden anxiety. Consuming fast-acting carbohydrates, such as fruit juice "
            "or glucose tablets, helps rapidly raise blood sugar levels. Following up with a small snack "
            "containing protein and complex carbohydrates can help prevent a recurrent drop in blood sugar. "
            "Frequent or severe hypoglycemic episodes should be evaluated by a healthcare professional to "
            "adjust treatment strategies safely."
        ),
    },
    # =========================================================================
    # Topic 2: Common Cold & Flu
    # =========================================================================
    {
        "topic": "Common Cold & Flu",
        "question": "How can one distinguish between the symptoms of a common cold and the flu?",
        "answer": (
            "A common cold typically causes mild symptoms such as a runny nose, sneezing, mild sore throat, "
            "and low-grade fever. In contrast, the flu usually presents abruptly with high fever, severe "
            "muscle aches, pronounced exhaustion, and a dry cough. Cold symptoms generally peak within a few "
            "days and resolve within a week, whereas flu symptoms may linger for two weeks or more. Seeking "
            "medical advice is recommended if severe chest pain, persistent high fever, or breathing "
            "difficulty occurs."
        ),
    },
    {
        "topic": "Common Cold & Flu",
        "question": "What home care measures can help relieve common cold symptoms?",
        "answer": (
            "Adequate bed rest and proper hydration are primary home care measures for recovering from a "
            "common cold. Drinking warm liquids such as herbal tea, clear broth, or warm water can soothe "
            "a sore throat and loosen respiratory secretions. Using a cool-mist humidifier or saline nasal "
            "spray may help relieve nasal congestion and airway dryness. If symptoms worsen significantly "
            "or fail to improve after ten days, consulting a healthcare practitioner is advisable."
        ),
    },
    {
        "topic": "Common Cold & Flu",
        "question": "Is it necessary to take antibiotics for a cold or influenza infection?",
        "answer": (
            "Antibiotics are designed to treat bacterial infections and are ineffective against viral "
            "illnesses such as colds and influenza. Using antibiotics for viral infections can cause "
            "unnecessary side effects and contributes to the global risk of antibiotic resistance. Most "
            "uncomplicated viral respiratory infections resolve naturally through rest, fluids, and self-care. "
            "A healthcare provider can evaluate your condition to determine whether secondary bacterial "
            "complications have developed."
        ),
    },
    {
        "topic": "Common Cold & Flu",
        "question": "What preventative measures reduce the risk of contracting respiratory viruses?",
        "answer": (
            "Washing hands frequently with soap and water for at least 20 seconds effectively reduces viral "
            "transmission. Avoiding close contact with individuals exhibiting cold or flu symptoms helps "
            "prevent the spread of respiratory pathogens. Refraining from touching your eyes, nose, and mouth "
            "minimizes the entry points for viral particles. Receiving an annual influenza vaccine is also "
            "recommended by public health guidance to reduce the risk of severe flu illness."
        ),
    },
    {
        "topic": "Common Cold & Flu",
        "question": "What warning signs indicate a cold or flu may be escalating into complications?",
        "answer": (
            "Warning signs of cold or flu complications include difficulty breathing, shortness of breath, "
            "persistent chest pain, and confusion. High fever lasting longer than three days or symptoms "
            "that improve initially but then return with worse fever and cough also indicate potential issues. "
            "Severe dizziness or inability to keep fluids down may signal dehydration requiring urgent care. "
            "If any of these alarming symptoms develop, immediate medical evaluation by a healthcare "
            "provider is strongly advised."
        ),
    },
    # =========================================================================
    # Topic 3: Hypertension
    # =========================================================================
    {
        "topic": "Hypertension",
        "question": "What is hypertension and why is it often called a silent condition?",
        "answer": (
            "Hypertension, commonly known as high blood pressure, occurs when the long-term force of blood "
            "against arterial walls is elevated. It is often referred to as a silent condition because many "
            "people experience no noticeable symptoms even when readings are dangerously high. Uncontrolled "
            "high blood pressure can gradually damage blood vessels, increasing the risk of heart disease, "
            "stroke, and kidney failure. Regular blood pressure screenings are essential for detecting the "
            "condition early and initiating proper clinical management."
        ),
    },
    {
        "topic": "Hypertension",
        "question": "What lifestyle modifications can assist in managing high blood pressure?",
        "answer": (
            "Adopting a heart-healthy diet low in sodium and rich in fruits, vegetables, and whole grains "
            "can support healthy blood pressure. Engaging in regular moderate aerobic exercise, such as brisk "
            "walking, helps strengthen the heart and lower vascular resistance. Limiting alcohol consumption, "
            "maintaining a healthy weight, and avoiding tobacco products further reduce cardiovascular strain. "
            "Individuals should consult a doctor before making major lifestyle modifications or adjusting "
            "prescribed antihypertensive care."
        ),
    },
    {
        "topic": "Hypertension",
        "question": "How is blood pressure measured and what do the numbers represent?",
        "answer": (
            "Blood pressure is recorded as two numbers measured in millimeters of mercury: systolic pressure "
            "over diastolic pressure. Systolic pressure represents the pressure in arteries when the heart "
            "contracts, while diastolic pressure measures pressure when the heart relaxes between beats. "
            "A reading consistently above standard reference thresholds indicates elevated blood pressure "
            "or hypertension. A qualified medical provider can properly interpret blood pressure trends and "
            "recommend appropriate follow-up testing."
        ),
    },
    {
        "topic": "Hypertension",
        "question": "Can chronic stress contribute to elevated blood pressure readings?",
        "answer": (
            "Chronic stress can lead to repeated spikes in blood pressure through the release of stress "
            "hormones like cortisol and adrenaline. Prolonged emotional strain may also foster indirect "
            "behaviors such as poor dietary choices, physical inactivity, or disrupted sleep patterns. "
            "Practicing stress-reduction techniques such as deep breathing, mindfulness, or light physical "
            "activity can help promote vascular health. If chronic stress impacts daily functioning or blood "
            "pressure control, discussing coping strategies with a healthcare provider is recommended."
        ),
    },
    {
        "topic": "Hypertension",
        "question": "When should an individual seek urgent care for high blood pressure?",
        "answer": (
            "An unusually high blood pressure reading accompanied by severe headache, chest pain, "
            "shortness of breath, or visual disturbances requires urgent care. Symptoms such as sudden weakness, "
            "numbness, or difficulty speaking alongside elevated blood pressure may indicate a cardiovascular "
            "or neurological emergency. It is unsafe to wait for severe hypertensive symptoms to subside "
            "without medical supervision. Immediate evaluation at an emergency department or urgent care "
            "facility is necessary to evaluate end-organ strain."
        ),
    },
    # =========================================================================
    # Topic 4: Headaches & Migraines
    # =========================================================================
    {
        "topic": "Headaches & Migraines",
        "question": "What are the primary differences between tension headaches and migraines?",
        "answer": (
            "Tension headaches typically present as a constant, dull ache or tight band-like pressure "
            "on both sides of the head. In contrast, migraines often cause moderate to severe throbbing pain "
            "on one side of the head and are frequently accompanied by nausea or vomiting. Migraines may "
            "also increase sensitivity to light and sound, making quiet, dark environments preferable during "
            "an episode. Consulting a healthcare professional can assist in differentiating between headache "
            "types and developing an appropriate care strategy."
        ),
    },
    {
        "topic": "Headaches & Migraines",
        "question": "What common triggers are associated with migraine attacks?",
        "answer": (
            "Common migraine triggers include psychological stress, irregular sleep schedules, skipped meals, "
            "and environmental changes. Certain foods, sensory inputs like bright lights or strong odors, and "
            "hormonal fluctuations can also precipitate migraine onset. Keeping a detailed headache diary "
            "can help individuals track potential triggers and identify recurring patterns over time. "
            "Discussing identified triggers with a medical doctor can inform lifestyle adjustments and "
            "preventive management plans."
        ),
    },
    {
        "topic": "Headaches & Migraines",
        "question": "What is a migraine aura and how does it manifest?",
        "answer": (
            "A migraine aura refers to temporary neurological symptoms that precede or accompany the onset "
            "of a migraine headache. Visual disturbances, such as flashing lights, flickering spots, or "
            "zigzag lines, are the most common form of migraine aura. Sensory aura symptoms may include "
            "tingling sensations in the face or hands, temporary speech difficulty, or mild numbness. "
            "First-time aura symptoms or sudden changes in aura patterns should always be evaluated by a "
            "healthcare practitioner to rule out other neurological conditions."
        ),
    },
    {
        "topic": "Headaches & Migraines",
        "question": "How can non-pharmacological methods assist in relieving headache discomfort?",
        "answer": (
            "Applying a cold or warm compress to the forehead or temples can help soothe localized headache "
            "discomfort. Resting in a quiet, darkened room away from bright screens and loud noises often "
            "provides relief during migraine episodes. Gentle neck stretches, staying well-hydrated, and "
            "practicing relaxation techniques may also alleviate muscular tension. If headaches occur "
            "frequently or interfere with daily activities, seeking guidance from a physician is recommended."
        ),
    },
    {
        "topic": "Headaches & Migraines",
        "question": "What red flag symptoms associated with headaches require immediate evaluation?",
        "answer": (
            "A sudden, explosive headache often described as the worst headache of your life requires "
            "immediate emergency evaluation. Headaches accompanied by high fever, stiff neck, confusion, "
            "seizures, or double vision also demand urgent medical attention. Headaches following a head injury "
            "or occurring with new weakness or numbness on one side of the body are additional red flags. "
            "Prompt emergency medical assessment is critical when experiencing any of these alarming "
            "neurological signs."
        ),
    },
    # =========================================================================
    # Topic 5: Sleep Issues
    # =========================================================================
    {
        "topic": "Sleep Issues",
        "question": "What is insomnia and what factors commonly contribute to it?",
        "answer": (
            "Insomnia is a sleep disorder characterized by persistent difficulty falling asleep, staying "
            "asleep, or experiencing non-restorative sleep. Common contributing factors include elevated "
            "stress levels, irregular sleep schedules, environmental disruptions, and poor sleep hygiene. "
            "Underlying medical conditions, chronic pain, or mental health concerns can also play a major role "
            "in sleep disturbances. If sleep difficulty persists for several weeks and impairs daytime focus, "
            "consulting a healthcare professional is advisable."
        ),
    },
    {
        "topic": "Sleep Issues",
        "question": "What practices define good sleep hygiene for improving sleep quality?",
        "answer": (
            "Good sleep hygiene involves maintaining a consistent sleep schedule by going to bed and waking "
            "up at the same times each day. Creating a quiet, dark, and cool bedroom environment helps signal "
            "to the body that it is time for rest. Avoiding electronic screens, caffeine, and heavy meals in "
            "the hours leading up to bedtime supports natural sleep architecture. Establishing a relaxing "
            "pre-sleep routine, such as reading or warm baths, can further improve sleep quality."
        ),
    },
    {
        "topic": "Sleep Issues",
        "question": "What are the signs and symptoms of obstructive sleep apnea?",
        "answer": (
            "Obstructive sleep apnea occurs when airway tissues collapse repeatedly during sleep, causing "
            "temporary breathing pauses. Common indicators include loud chronic snoring, choking or gasping "
            "sounds during sleep, and excessive daytime drowsiness. Waking up with a dry mouth, morning "
            "headaches, or difficulty concentrating during the day are additional frequent signs. A medical "
            "evaluation by a sleep specialist is necessary to accurately diagnose and manage sleep apnea effectively."
        ),
    },
    {
        "topic": "Sleep Issues",
        "question": "How does nighttime exposure to electronic devices affect sleep cycles?",
        "answer": (
            "Electronic devices emit blue light, which can suppress the brain's natural secretion of "
            "melatonin, the hormone regulating sleep cycles. Exposure to bright screens near bedtime can "
            "delay sleep onset and disrupt circadian rhythms essential for deep rest. Engaging with mentally "
            "stimulating content on digital devices can also heighten alertness when the body needs relaxation. "
            "Dimming screen brightness or turning off digital devices at least one hour before bed helps "
            "encourage restorative sleep."
        ),
    },
    {
        "topic": "Sleep Issues",
        "question": "When should an individual consult a healthcare professional about sleep disturbances?",
        "answer": (
            "You should consult a healthcare provider if sleep disruptions persist for several weeks or cause "
            "severe daytime fatigue and mood changes. Chronic sleep issues can negatively affect cardiovascular "
            "health, immune function, and cognitive performance over time. A physician can evaluate whether an "
            "underlying physical condition, sleep disorder, or medication side effect is responsible. Seeking "
            "professional medical advice helps identify appropriate treatment options and tailored management strategies."
        ),
    },
]
