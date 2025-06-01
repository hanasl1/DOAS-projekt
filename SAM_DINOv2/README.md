# Uvod

Alternativni pristup koji se oslanja na snagu modernih samonadziranih vizualnih transformera (ViT), poput DINOv2 i modela za segmentaciju, poput SAM-a za postizanje preciznog podudaranja objekta s upita na dio slike scene.

# Pristup: Kombinacija DINOv2 Embedinga i SAM Segmentacije

Naš primarni pristup za podudaranje objekta iz upita (eng. query image) na dio slike scene (eng. scene/composite image) temelji se na dva napredna modela dubokog učenja: DINOv2 za generiranje vizualnih reprezentacija (eng. embeddings) i Segment Anything Model (SAM) za preciznu segmentaciju objekata u sceni. Proces se može sažeti u sljedeće korake:

1. Generiranje reprezentacije upita: Ulazna slika objekta (upit) prolazi kroz DINOv2 model kako bi se generirao njen globalni embeding – kompaktni numerički vektor koji sažima semantičke i vizualne karakteristike objekta.
2. Segmentacija scene: Slika scene se obrađuje pomoću SAM modela. SAM generira skup maski koje predstavljaju potencijalne objekte ili distinktivne regije unutar scene.
3. Generiranje reprezentacije Segmenata Scene: Za svaku masku generiranu od strane SAM-a, odgovarajući dio slike (eng. bounding box) scene se izdvaja. Ovaj izolirani segment zatim prolazi kroz DINOv2 model kako bi se dobila njegova reprezentacija.
4. Usporedba reprezentacija: Embedding objekta s upita uspoređuje se sa svakim embeddingom generiranim iz segmenata scene koristeći mjeru kosinusne sličnosti.
5. Identifikacija Podudaranja: Segment scene čiji je embedding najsličniji embeddingu upita (i prelazi određeni prag sličnosti) proglašava se podudaranjem. Granični okvir (eng. bounding box) tog segmenta definira lokaciju objekta u sceni.

Kao alternativu ili komplementarnu metodu, također je implementiran pristup temeljen isključivo na DINOv2 modelu, gdje se umjesto SAM segmentacije koristi strategija generiranja i usporedbe preklapajućih pravokutnih isječaka (eng. patches) scene s upitom, uz primjenu Non-Maximum Suppression (NMS) tehnike za filtriranje redundantnih detekcija. Ovo je manje efikasna metoda koja daje lošije rezultate.

# Detalji Modela

DINOv2 (Self DIstillation with NO labels v2): DINOv2 je vizualni transformer (ViT) model koji je treniran samonadziranom metodom na ogromnim skupovima neoznačenih slika. Ključna prednost DINOv2 modela je njegova sposobnost učenja vizualnih značajki koje su semantički bogate i robusne na različite transformacije poput promjene osvjetljenja, djelomične okluzije i promjene perspektive. Za razliku od SIFT-a, koji se fokusira na lokalne gradijente i teksture oko ključnih točaka, DINOv2 generira globalni ili poluglobalni "embedding" cijele slike ili isječka.

SAM (Segment Anything Model): SAM je model za segmentaciju slika. Njegova revolucionarnost leži u sposobnosti generiranja visokokvalitetnih maski za objekte na slici na temelju različitih vrsta upita (eng. prompts), ili čak bez eksplicitnog upita ("segment everything" mod). U našem pristupu, koristimo SamAutomaticMaskGenerator koji automatski identificira i segmentira sve uočljive objekte ili regije na slici scene. Ovo nam omogućuje da izoliramo potencijalne kandidate za podudaranje od pozadine i drugih objekata, što može dovesti do "čišćih" i diskriminativnijih embedinga kada se ti segmenti proslijede DINOv2 modelu. SAM koristi enkoder slike (obično ViT) i lagani dekoder maski. Za naš rad, koristili smo vit_b varijantu SAM modela zbog balansa između performansi i računalnih zahtjeva.

# Implementacijski Detalji i Metrike Evaluacije

Naša implementacija u Pythonu koristi PyTorch kao osnovni radni okvir. Za DINOv2 i SAM modele oslanjamo se na službene implementacije i predtrenirane težine dostupne putem Hugging Face i segment_anything biblioteka. Usporedba embedinga vrši se pomoću kosinusne sličnosti, a kao prag za prihvaćanje podudaranja koristi se empirijski određena vrijednost, oovisno o metodi i specifičnosti objekata.

Za evaluaciju performansi našeg pristupa, kao i za usporedbu s tradicionalnim (SIFT) i drugim dubinskim metodama (SuperPoint+SuperGlue), koristimo sljedeće metrike:

- vrijeme izvođenja (generalno sporije)
- uspjeh u pronalasku dovoljno sličnog kandidata
- kosinusna sličnost pronađenog segmenta i query objekta

# Rezultati

// uglavnom uspije pronaći objekt, ovisno o pragu sličnosti
// često bounding box obuhvaća i objekt koji je iznad traženog
// dogodi se i da spoji na krivi objekt u slučaju kad je objekt jako natkriven
// s obzirom da su tu i semantičke značajke, ne gleda samo npr. boju i oblik nego i "što" je taj objekt - dobro kad su kategorije / query objekti jako različiti

# Zaključak

Zaključno, kombinacija DINOv2 i SAM modela nudi obećavajući, moderan pristup rješavanju problema podudaranja objekata u izazovnim uvjetima, predstavljajući alternativu tradicionalnim i drugim metodama temeljenim na dubokom učenju.
Bolji rezultati mogli bi se postići:

- dotreniranjem modela za detekciju i segmentaciju na većem podatkovnom skupu specifičnom za problem.
- metodom sužavanja segmenata koje SAM pronađe kao kandidate maksimizacijom kosinusne sličnosti