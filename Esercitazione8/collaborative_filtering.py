
#!/usr/bin/env python
# Implementation of collaborative filtering recommendation engine


from recommendation_data import dataset
from math import sqrt

def similarity_score(person1,person2):
	
	# Returns ratio Euclidean distance score of person1 and person2 

	both_viewed = {}		# To get both rated items by person1 and person2

	for item in dataset[person1]:
		if item in dataset[person2]:
			both_viewed[item] = 1

		# Conditions to check they both have an common rating items	
		if len(both_viewed) == 0:
			return 0

		# Finding Euclidean distance 
		sum_of_eclidean_distance = []	

		for item in dataset[person1]:
			if item in dataset[person2]:
				sum_of_eclidean_distance.append(pow(dataset[person1][item] - dataset[person2][item],2))
		sum_of_eclidean_distance = sum(sum_of_eclidean_distance)

		return 1/(1+sqrt(sum_of_eclidean_distance))

def pearson_correlation(person1,person2):

	# To get both rated items
	both_rated = {}
	for item in dataset[person1]:
		if item in dataset[person2]:
			both_rated[item] = 1

	number_of_ratings = len(both_rated)		
	
	# Checking for number of ratings in common
	if number_of_ratings == 0:
		return 0

	# Add up all the preferences of each user
	person1_preferences_sum = sum([dataset[person1][item] for item in both_rated])
	person2_preferences_sum = sum([dataset[person2][item] for item in both_rated])

	# Sum up the squares of preferences of each user
	person1_square_preferences_sum = sum([pow(dataset[person1][item],2) for item in both_rated])
	person2_square_preferences_sum = sum([pow(dataset[person2][item],2) for item in both_rated])

	# Sum up the product value of both preferences for each item
	product_sum_of_both_users = sum([dataset[person1][item] * dataset[person2][item] for item in both_rated])

	# Calculate the pearson score
	numerator_value = product_sum_of_both_users - (person1_preferences_sum*person2_preferences_sum/number_of_ratings)
	denominator_value = sqrt((person1_square_preferences_sum - pow(person1_preferences_sum,2)/number_of_ratings) * (person2_square_preferences_sum -pow(person2_preferences_sum,2)/number_of_ratings))
	if denominator_value == 0:
		return 0
	else:
		r = numerator_value/denominator_value
		return r 

def most_similar_users(person,number_of_users):
	# returns the number_of_users (similar persons) for a given specific person.
	scores = [(pearson_correlation(person,other_person),other_person) for other_person in dataset if  other_person != person ]
	
	# Sort the similar persons so that highest scores person will appear at the first
	scores.sort()
	scores.reverse()
	return scores[0:number_of_users]

def user_recommendations(person, function):

	# Gets recommendations for a person by using a weighted average of every other user's rankings
	totals = {}
	simSums = {}
	rankings_list = []

	if function == "pearson_with_approximation":
		recommendations_approximation()

	for other in dataset:
		# don't compare me to myself
		if other == person:
			continue

		# scelta della funzione di similarity
		if function == "pearson":
			sim = pearson_correlation(person, other)
		elif function == "euclidea":
			sim = similarity_score(person, other)
		elif function == "pearson_with_approximation":
			sim = similarity_score(person, other)

		# ignore scores of zero or lower
		if sim <=0: 
			continue

		for item in dataset[other]:

			# only score movies i haven't seen yet
			if item not in dataset[person] or dataset[person][item] == 0:

				# Similrity * score
				totals.setdefault(item,0)
				totals[item] += dataset[other][item]* sim

				# sum of similarities
				simSums.setdefault(item,0)
				simSums[item] += sim

		# Stampa Informazioni
		print(f"\n{other} Similarity: {round(sim,2)}")
		for item in dataset[other]:
			if item not in dataset[person]:
				print(f"Film: {item} - Voto: {round(dataset[other][item],2)} - Film Similarity: {round(dataset[other][item] * sim, 2)}")

	# Create the normalized list
	rankings = [(total/simSums[item],item) for item,total in totals.items()]
	rankings.sort()
	rankings.reverse()

	# returns the recommended items
	recommendataions_list = [recommend_item for score,recommend_item in rankings]
	return recommendataions_list
		
def calculateSimilarItems(prefs,n=10):
        # Create a dictionary of items showing which other items they
        # are most similar to.
        result={}
        # Invert the preference matrix to be item-centric
        itemPrefs=transformPrefs(prefs)
        c=0
        for item in itemPrefs:
                # Status updates for large datasets
                c+=1
                if c%100==0: print("%d / %d" % (c,len(itemPrefs)))
                # Find the most similar items to this one
                scores=topMatches(itemPrefs,item,n=n,similarity=sim_distance)
                result[item]=scores
        return result

def recommendations_approximation():

	# Per ogni utente k
	for user in dataset:
		average_vote = 0.0
		average_local = 0.0
		sum_sims = 0.0
		sum_sims_weighted = 0.0
		
		print(f"\nUtente: {user}")

		# Calcolo il suo voto medio
		for item in dataset[user]:
			average_vote += dataset[user][item]
		average_vote /= len(dataset[user])

		print(f"Voto medio: {round(average_vote,2)}")
		
		# Per ogni film i che ha votato l'utente k
		for item in dataset[user]:
			for other in dataset:
				
				# saltiamo l'utente stesso
				if other == user:
					continue

				# consideriamo solo gli altri utenti che hanno votato lo stesso film
				if item in dataset[other]:

					# ne calcoliaml la similarità
					sim = pearson_correlation(user, other)
					
					# calcolo la media di questo utente
					for other_item in dataset[other]:
						average_local += dataset[other][other_item]
					average_local /= len(dataset[other])

					# aggiorne le somme delle simillarità
					sum_sims_weighted += sim * (dataset[other][other_item] - average_local)
					sum_sims += sim

			# aggiorno la raccomandazione dell'utente k sul prodotto i
			dataset[user][item] = average_vote + (sum_sims_weighted / sum_sims)

			print(f"Raccomandazione per il prodotto {item}: {round(dataset[user][item],2)}")

if __name__ == "__main__" :
	print("\nMisura di Similarità: Correlazione di Pearson")
	print(f"\nFilm consigliati per Toby: {user_recommendations('Toby','pearson')}")

	print("\nMisura di Similarità: Distanza Euclidea")
	print(f"\nFilm consigliati per Toby: {user_recommendations('Toby','euclidea')}")

	print("\nMisura di Similarità: Correlazione di Pearson con Approssimaizone")
	print(f"\nFilm consigliati a Toby: {user_recommendations('Toby','pearson_with_approximation')}")

	


