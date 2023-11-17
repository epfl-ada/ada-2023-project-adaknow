DATASET_PATHS = {
    'movie_meta': './data/movie.metadata.tsv',
    'char_meta': './data/character.metadata.tsv',
    'plot_sum': './data/plot_summaries.txt',
    'film_tropes' : './data/film_tropes.csv',
    'female_tvtropes' : './data/female_tvtropes.csv',
    'male_tvtropes' : './data/male_tvtropes.csv',
    'unisex_tvtropes' : './data/unisex_tvtropes.csv'

}

COL_NAMES = {
    'movie_meta': [
        "Wikipedia movie ID", 
        "Freebase movie ID", 
        "Movie name", 
        "Movie release date",
        "Movie box office revenue", 
        "Movie runtime", 
        "Languages", 
        "Countries", 
        "Genres"
    ],

    'char_meta': [
        "Wikipedia movie ID",
        "Freebase movie ID",
        "Movie release date",
        "Character name",
        "Actor date of birth",
        "Actor gender",
        "Actor height (in meters)",
        "Actor ethnicity (Freebase ID)",
        "Actor name",
        "Actor age at movie release",
        "Freebase character/actor map ID",
        "Freebase character ID",
        "Freebase actor ID"
    ],

    'plot_sum': [
        "Wikipedia movie ID",
        "Plot summary"
    ],
}

GENRE_REDUCTION_MAP = {
    "Softcore Porn": ["Adult"],
    "Prison film": ["Prison"],
    "Pinku eiga": ["Adult", "Japanese", "Asian"],
    "Inventions & Innovations": None, 
    "Bruceploitation": ["Martial Arts", "Exploitation", "Low Budget"],
    "Tollywood": ["Indian", "Asian"],
    "Instrumental Music": ["Music/Dance"],
    "Splatter film": ["Horror"],
    "Female buddy film": ["Adventure", "Comedy", "Women"],
    "Detective fiction": ["Crime", "Thriller"],                 # 10
    "Dystopia": ["Dystopia"],
    "Black comedy": ["Comedy"],
    "History": ["History"],
    "Silhouette animation": ["Animation"],
    "Combat Films": ["Action"],
    "Educational": ["Educational"],
    "Heaven-Can-Wait Fantasies": None,
    "Outlaw": ["Crime"],
    "Marriage Drama": ["Drama", "Women"],
    "Health & Fitness": ["Sport"],                              # 20
    "Stand-up comedy": ["Comedy"],
    "Sci-Fi Adventure": ["Science Fiction", "Adventure"],
    "Road-Horror": ["Horror"],
    "Anthology": ["Anthology"],
    "Biography": ["Biography"],
    "Doomsday film": ["Dystopia"],
    "Malayalam Cinema": ["Indian"],
    "Musical comedy": ["Musical", "Comedy"],
    "Chick flick": ["Romance", "Teen", "Women"],
    "Action/Adventure": ["Action", "Adventure"],                # 30
    "Horror Comedy": ["Horror", "Comedy"],
    "Spaghetti Western": ["Western", "European"],
    "Crime Drama": ["Crime", "Drama"],
    "Monster": ["Horror"],
    "Kafkaesque": ["Dystopian", "Psychological"],
    "Filipino": ["Filipino", "Asian"],
    "Movie serial": ["Action"],
    "War effort": ["Politics", "War"],
    "Extreme Sports": ["Sports"],
    "Sex comedy": ["Adult", "Comedy"],                        # 40
    "Documentary": ["Documentary"], 
    "B-Western": ["Western", "Low Budget"],
    "Existentialism": ["Philosophical"],
    "Demonic child": ["Horror"],
    "Buddy film": ["Adventure", "Comedy"],
    "Early Black Cinema": ["Black Cinema"],
    "Therimin music": ["Music/Dance"],
    "Coming-of-age film": ["Social"],
    "Sexploitation": ["Adult", "Exploitation", "Low Budget"],
    "Outlaw biker film": ["Crime"],                         # 50
    "Gross out": ["Horror"],
    "Drama": ["Drama"],
    "Neorealism": ["Art Film"],
    "Christmas movie": ["Holidays"],
    "British New Wave": ["Art Film", "European"],
    "Comedy of Errors": ["Comedy"],
    "Computer Animation": ["Animation"],
    "Children's/Family": ["Kids", "Family"],
    "Surrealism": ["Art Film"],
    "Suspense": ["Thriller"],                               # 60
    "Dogme 95": ["Avant-Garde"],
    "Family Film": ["Family"],
    "Stop motion": ["Stop Motion"],
    "Mockumentary": ["Documentary"],
    "Ealing Comedies": ["Comedy"],
    "Czechoslovak New Wave": ["Art Film", "European"],
    "Archives and records": ["History"],
    "Social issues": ["Social"],
    "Environmental Science": ["Nature", "Educational"],
    "Short Film": ["Short Film"],                           # 70
    "Religious Film": ["Religious"],
    "The Netherlands in World War II": ["History", "War"],
    "Propaganda film": ["Politics"],
    "Historical Epic": ["History", "Epic"],
    "Action": ["Action"],
    "Horror": ["Horror"],
    "C-Movie": ["Low Budget"],
    "Film-Opera": ["Music/Dance"],
    "Period piece": ["History", "Drama"],
    "Silent film": ["Silent"],                              # 80
    "Political thriller": ["Politics", "Thriller"],
    "Absurdism": ["Art Film", "Philosophical"],
    "Gulf War": ["History", "War"],
    "Humour": ["Comedy"],
    "Sports": ["Sports"],
    "Bollywood": ["Indian"],
    "Historical Documentaries": ["History", "Documentary"],
    "Tokusatsu": ["Japanese"],
    "Road movie": ["Adventure"],
    "Conspiracy fiction": ["Politics", "Fiction"],          # 90
    "Punk rock": ["Music/Dance"],
    "Singing cowboy": ["Music/Dance", "Western"],
    "Breakdance": ["Music/Dance"],
    "Fictional film": ["Fiction"],
    "Feature film": None,
    "Epic": ["Epic"],
    "Journalism": ["Politics"],
    "Buddy Picture": ["Adventure", "Comedy"],
    "Children's Issues": ["Social"],
    "Family-Oriented Adventure": ["Family", "Adventure"],   # 100
    "Psychological thriller": ["Psychological", "Thriller"],
    "Children's": ["Kids"],
    "Z movie": ["Low Budget"],
    "Statutory rape": ["Adult"],
    "Mumblecore": ["Low Budget"],
    "Nature": ["Nature"],
    "Teen": ["Teen"],
    "Superhero movie": ["Action", "Science Fiction"],
    "Biographical film": ["Biography"],
    "British Empire Film": ["History", "War"],              # 110
    "Experimental film": ["Avant-Garde"],
    "Courtroom Drama": ["Drama"],
    "Comedy of manners": ["Comedy"],
    "Homoeroticism": ["LGBTQ", "Adult"],
    "Musical Drama": ["Musical", "Drama"],
    "Backstage Musical": ["Music/Dance"],
    "Screwball comedy": ["Romance", "Comedy"],
    "Libraries and librarians": None,
    "Erotica": ["Adult"],
    "Plague": ["Dystopian"],                                # 120
    "Martial Arts Film": ["Martial Arts"],
    "Slice of life story": ["Biography"],
    "Satire": ["Comedy"],
    "Animal Picture": ["Animation"],
    "LGBT": ["LGBTQ"],
    "Film adaptation": ["Film Adaptation"],
    "Alien Film": ["Horror"],
    "Boxing": ["Sports"],
    "Vampire movies": ["Horror"],
    "Parkour in popular culture": ["Sports"],               # 130
    "Whodunit": ["Crime"],
    "Revisionist Western": ["Western"],
    "World History": ["History"],
    "News": ["Politics"],
    "Political satire": ["Politics", "Comedy"],
    "Costume Horror": ["Horror"],
    "Linguistics": None,
    "Alien invasion": ["Horror"],
    "Indie": ["Independent"],
    "Master Criminal Films": ["Crime"],                     # 140
    "Anime": ["Animated", "Japanese"],
    "Haunted House Film": ["Horror"],
    "Baseball": ["Sport"],
    "Erotic thriller": ["Adult", "Thriller"],
    "B-movie": ["Low Budget"],
    "Foreign legion": ["War", "History"],
    "Biker Film": ["Sport"],
    "Political Documetary": ["Politics", "Documentary"],
    "Social problem film": ["Social"],
    "Detective": ["Crime"],                                 # 150
    "Blaxploitation": ["Black Cinema", "Exploitation", "Low Budget"],
    "Gangster Film": ["Crime"],
    "Education": ["Educational"],
    "Romantic drama": ["Romance", "Drama", "Women"],
    "Auto racing": ["Cars"],
    "Gender Issues": ["Gender Issues", "Social"],
    "Period Horror": ["Horror"],
    "Inspirational Drama": ["Drama"],
    "Fantasy": ["Fantasy"],
    "Airplanes and airports": None,                         # 160
    "Swashbuckler films": ["Action"],
    "Time travel": ["Science Fiction"],
    "Romantic fantasy": ["Romance", "Fantasy"],
    "Beach Party film": ["Beach"],
    "Comedy-drama": ["Comedy", "Drama"],
    "Americana": ["USA"],
    "Business": None,
    "Illnesses & Disabilities": ["Social", "Medical"],
    "Nuclear warfare": ["Dystopian", "War"],
    "Crime Thriller": ["Crime", "Thriller"],                # 170
    "Steampunk": ["Science Fiction", "Futuristic"],
    "Children's Entertainment": ["Kids"],
    "Anti-war": ["Social", "War"],
    "Star vehicle": ["Star Vehicle"],
    "Feminist Film": ["Feminism"],
    "Gay Themed": ["LGBTQ"],
    "World cinema": ["World Cinema"],
    "Chase Movie": ["Action", "Cars"],
    "Hip hop movies": ["Music/Dance", "Black Cinema"],
    "Gothic Film": ["Horror", "Art Film"],                  # 180
    "Children's Fantasy": ["Fantasy", "Kids"],
    "Film noir": ["Crime", "Drama"],
    "Romantic comedy": ["Romance", "Comedy"],
    "Western": ["Western"],
    "Caper story": ["Crime"],
    "Samurai cinema": ["Action", "Japanese"],
    "Animated Musical": ["Animated", "Music/Dance"],
    "Cult": ["Religious"],
    "Science Fiction": ["Science Fiction"],
    "Crime Fiction": ["Crime", "Fiction"],
    "Disaster": ["Dystopian"],                            # 190
    "Indian Western": ["Indian"],
    "Black-and-white": ["Black and White"],
    "Sponsored film": ["Sponsored"],
    "Psychological horror": ["Psychological", "Horror"],
    "Television movie": ["Television"],
    "Legal drama": ["Social", "Drama"],
    "Addiction Drama": ["Drama"],
    "Fan film": None,
    "Culture & Society": ["Social"],                  # 200
    "Sci Fi Pictures original films": ["Science Fiction"],
    "Heist": ["Crime"],
    "Docudrama": ["Documentary", "Drama"],
    "Buddy cop": ["Crime", "Adventure"],
    "Spy": ["Crime"],
    "Animation": ["Animation"],
    "Horse racing": ["Sports"],
    "Superhero": ["Action", "Science Fiction"],
    "Cold War": ["History"],
    "Dance": ["Music/Dance"],                      # 210
    "Goat gland": None,
    "Tragedy": ["Drama"],
    "Finance & Investing": None,
    "Film": None,
    "Anti-war film": ["Social", "War"],
    "Graphic & Applied Arts": None,
    "Pornography": ["Adult"],
    "Escape Film": ["Prison"],
    "Language & Literature": ["Literary"],
    "Bengali Cinema": ["Social", "Indian", "Asian"],                 # 220
    "Essay Film": ["Literary"],
    "Family & Personal Relationships": ["Social"],
    "Private military company": None,
    "Crime Comedy": ["Crime", "Comedy"],
    "Mystery": ["Thriller"],
    "Zombie Film": ["Horror", "Dystopian"],
    "Concert film": ["Music/Dance"],
    "Media Studies": ["Social"],
    "Anthropology": ["Social"],
    "Hybrid Western": ["Western"],
    "Heavenly Comedy": ["Comedy"],
    "Ensemble Film": ["Ensemble"],
    "Comedy": ["Comedy"],
    "Women in prison films": ["Prison"],
    "Space western": ["Western", "Science Fiction"],
    "Thriller": ["Thriller"],
    "Workplace Comedy": ["Comedy"],
    "Gay": ["LGBTQ"],
    "Animated cartoon": ["Animated"],
    "Fantasy Adventure": ["Fantasy", "Adventure"],
    "Music": ["Music/Dance"],
    "Supermarionation": None,
    "Adventure Comedy": ["Adventure", "Comedy"],
    "Art film": ["Art Film"],
    "Science fiction Western": ["Western", "Science Fiction"],
    "Sword and sorcery films": ["Science Fiction"],
    "Historical fiction": ["History", "Fiction"],
    "Glamorized Spy Film": ["Crime"],
    "Future noir": ["Crime", "Drama", "Science Fiction"],
    "Christian film": ["Religious"],
    "Childhood Drama": ["Drama"],
    "Erotic Drama": ["Adult", "Drama"],
    "Kitchen sink realism": ["Art Film", "European"],
    "Fantasy Comedy": ["Fantasy", "Comedy"],
    "Clay animation": ["Stop Motion"],
    "Romantic thriller": ["Romance", "Thriller"],
    "Gay pornography": ["Adult", "LGBTQ"],
    "Filipino Movies": ["Filipino", "Asian"],
    "Comedy Thriller": ["Comedy", "Thriller"],
    "Biopic [feature]": ["Biography"],
    "Ninja movie": ["Action"],
    "Comedy Western": ["Comedy", "Western"],
    "Epic Western": ["Epic", "Western"],
    "Wuxia": ["Martial Arts", "Chinese"],
    "Prison escape": ["Prison"],
    "Tamil cinema": ["Indian"],
    "Pornographic movie": ["Adult"],
    "Media Satire": ["Politics", "Comedy"],
    "Supernatural": ["Science Fiction"],
    "Albino bias": None,
    "Movies About Gladiators": ["Action", "History"],
    "Action Comedy": ["Action", "Comedy"],
    "Werewolf fiction": ["Horror", "Fiction"],
    "Space opera": ["Science Fiction"],
    "Latino": ["Latino"],
    "Reboot": ["Reboot"],
    "Revisionist Fairy Tale": None,
    "Animals": ["Nature"],
    "Albino bias": None,
    "Action Comedy": ["Action", "Comedy"],
    "Werewolf fiction": ["Horror", "Fiction"],
    "Space opera": ["Science Fiction"],
    "Reboot": ["Reboot"],
    "Revisionist Fairy Tale": None,
    "Animals": ["Nature"],
    "Sci-Fi Thriller": ["Science Fiction", "Thriller"],
    "Political drama": ["Politics", "Drama"],
    "Mondo film": ["Documentary", "Horror"],
    "Costume Adventure": ["Adventure"],
    "Giallo": ["Thriller", "Psychological", "Horror", "European"],
    "Sword and Sandal": ["History"],
    "Adult": ["Adult"],
    "Expressionism": ["Art Film"],
    "Gross-out film": ["Horror"],
    "Fairy tale": ["Science Fiction"],
    "Tragicomedy": ["Drama", "Comedy"],
    "Point of view shot": None,
    "Chinese Movies": ["Chinese"],
    "Slapstick": ["Comedy"],
    "Northern": ["Western"],
    "Sword and sorcery": ["Science Fiction", "Adventure"],
    "Prison": ["Prison"],
    "Courtroom Comedy": ["Social", "Comedy"],
    "Coming of age": ["Social"],
    "School story": ["Social"],
    "New Queer Cinema": ["LGBTQ"],
    "Comdedy": ["Comedy"],
    "Exploitation": ["Exploitation", "Low Budget"],
    "Crime": ["Crime"],
    "Revenge": ["Drama"],
    "Costume drama": ["Drama"],
    "Medical fiction": ["Medical", "Fiction"],
    "Avant-garde": ["Avant-Garde"],
    "Parody": ["Comedy"],
    "Action Thrillers": ["Action", "Thrillers"],
    "Apocalyptic and post-apocalyptic fiction": ["Dystopian", "Fiction"],
    "Hagiography": ["Religious"],
    "Creature Film": ["Horror"],
    "New Hollywood": ["USA"],
    "Adventure": ["Adventure"],
    "Jukebox musical": ["Music/Dance"],
    "Interpersonal Relationships": ["Social"],
    "Acid western": ["Western"],
    "Law & Crime": ["Social", "Crime"],
    "Juvenile Delinquency Film": ["Crime"],
    "Race movie": ["Cars"],
    "Natural disaster": ["Nature"],
    "Slasher": ["Horror"],
    "Live action": ["Animation"],
    "Monster movie": ["Horror"],
    "Jungle Film": ["Nature"],
    "Bloopers & Candid Camera": None,
    "Sci-Fi Horror": ["Science Fiction", "Horror"],
    "Operetta": ["Music/Dance", "Art Film"],
    "Fantasy Drama": ["Fantasy", "Drama"],
    "Melodrama": ["Drama"],
    "Stoner film": ["Comedy"],
    "Psycho-biddy": ["Horror", "Thriller"],
    "Rockumentary": ["Music/Dance", "Documentary"],
    "War film": ["War"],
    "Archaeology": None,
    "Holiday Film": ["Holidays"],
    "Filmed Play": None,
    "Family Drama": ["Drama"],
    "Natural horror films": ["Nature", "Horror"],
    "Musical": ["Music/Dance"],
    "Political cinema": ["Politics"],
    "Cyberpunk": ["Science Fiction", "Dystopian", "Futuristic"],
    "Romance Film": ["Romance"],
    "Comedy horror": ["Comedy", "Horror"],
    "Roadshow/Carny": ["Art Film"],
    "Remake": ["Remake"],
    "Domestic Comedy": ["Comedy"],
    "Beach Film": ["Beach"],
    "Pre-Code": ["USA"],
    "Roadshow theatrical release": None,
    "Comedy film": ["Comedy"],
    "Travel": None,
    "Computers": None,
    "Historical drama": ["History", "Drama"],
    "Film & Television History": ["History"],
    "Neo-noir": ["Crime", "Drama"],
    "Gay Interest": ["LBGTQ"],
    "Hardcore pornography": ["Adult"],
    "Mythological Fantasy": ["Fantasy", "Fiction"],
    "Camp": ["Art Film"],
    "Film à clef": ["Fiction"],
    "Patriotic film": ["USA", "Propaganda"],
    "Japanese Movies": ["Japanese", "Asian"],
    "Cavalry Film": ["Western", "USA"],
}