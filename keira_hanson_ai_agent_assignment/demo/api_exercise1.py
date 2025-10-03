import requests
import json
import logging
import os

#  Configuration 
logging.basicConfig(
   level=logging.INFO,   # Change to DEBUG for more details
   format="%(asctime)s - %(levelname)s - %(message)s"
   )

API_URL = "https://catfact.ninja/fact"
OUTPUT_FILE = "cat_facts.json"


#  Functions 
def get_cat_fact():
   """Fetch a single cat fact from the API."""
   try:
      response = requests.get(API_URL, timeout=5)   # add timeout
      response.raise_for_status()   # raise error if status != 200
   
      data = response.json()
      fact = data.get("fact")
   
      if fact:
         return fact
      else:
         logging.error("Response JSON missing 'fact': %s", data)
         return None
   
   except requests.exceptions.Timeout:
      logging.error("Request timed out.")
   except requests.exceptions.RequestException as e:
      logging.error("Network/HTTP error: %s", e)
   except json.JSONDecodeError as e:
      logging.error("Failed to parse JSON: %s", e)
   except Exception as e:
      logging.error("Unexpected error: %s", e)

   return None


def get_multiple_cat_facts(count=5):
   """Fetch multiple cat facts (may contain duplicates)."""
   facts = []
   attempts = 0

   while len(facts) < count and attempts < count * 3:  # retry limit
      fact = get_cat_fact()
      attempts += 1
      if fact and fact not in facts:   # avoid duplicates
         facts.append(fact)
         logging.info("Got cat fact: %s", fact)
      elif fact:
         logging.debug("Duplicate fact, skipping.")

   if len(facts) < count:
      logging.warning("Only retrieved %d unique facts.", len(facts))

   return facts


def save_facts_to_file(facts, filename=OUTPUT_FILE):
   """Save facts to a JSON file."""
   try:
      with open(filename, "w", encoding="utf-8") as f:
         json.dump(facts, f, indent=4, ensure_ascii=False)
      full_path = os.path.abspath(filename)
      logging.info("Saved %d facts to %s", len(facts), full_path)
      print(f"\nJSON file saved at: {full_path}")
   except Exception as e:
      logging.error("Failed to save facts: %s", e)


# Main 
if __name__ == "__main__":
   logging.info("Starting cat fact retrieval...")

   cat_facts = get_multiple_cat_facts(5)
   print("\n Cat Facts:")
   for i, fact in enumerate(cat_facts, start=1):
      print(f"{i}. {fact}")

   save_facts_to_file(cat_facts)

   logging.info("Process complete.")




