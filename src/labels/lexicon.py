"""Léxico multilingüe de conceptos para el extractor de reglas.

Idiomas: en, es, de, hr, el, bg, tr, nl, fr. Cada entrada es un fragmento de regex que se
normaliza igual que el texto (minúsculas, sin diacríticos, ı->i, ß->ss), así que se puede
escribir con o sin acentos. Por defecto cada entrada exige frontera de palabra al inicio
(funciona como raíz: "menisc" encuentra "meniscus" y "meniscos"). Si empieza con "*" se
busca también dentro de palabras compuestas (alemán y neerlandés: "gelenkerguss").

No usar escapes en mayúscula (\\W, \\S, \\D): la normalización los pasaría a minúscula.
"""

# ---------- anatomía ----------
ACL = [
    r"acl\b", r"anterior cruciate", r"lca\b", r"cruzado anterior", r"croise anterieur",
    r"vkb\b", r"vorder\w* kreuzband", r"*vorderes? kreuzband", r"prednj\w* kri[zž]\w* lig",
    r"προσθι\w* χιαστ", r"πχσ\b", r"предн\w* кръстн", r"on capraz bag", r"ocb\b",
    r"voorste kruisband", r"*voorste kruisband",
]
PCL = [r"pcl\b", r"posterior cruciate", r"lcp\b", r"cruzado posterior", r"hkb\b", r"hinter\w* kreuzband",
       r"straznj\w* kri", r"οπισθι\w* χιαστ", r"задн\w* кръстн", r"arka capraz", r"achterste kruisband",
       r"croise posterieur"]
CRUCIATE_GENERIC = [
    r"cruciate ligaments", r"cruciates", r"ligamentos cruzados", r"cruzados", r"*kreuzbander",
    r"kri[zž]ni ligamenti", r"χιαστ\w* συνδεσμ\w*", r"χιαστων", r"кръстни връзки", r"кръстните връзки",
    r"capraz baglar", r"*kruisbanden", r"ligaments croises",
]
MCL = [
    r"mcl\b", r"medial collateral", r"tibial collateral", r"colateral medial", r"colateral interno",
    r"lcm\b", r"lli\b", r"*innenband", r"medial\w* (seiten|kollateral)band", r"medijaln\w* kolateraln",
    r"εσω πλαγι\w* συνδεσμ", r"εσω πλαγι", r"медиал\w* колатерал", r"вътрешн\w* странич",
    r"medial kollateral", r"mkl\b", r"mediale collaterale", r"*binnenband", r"mediaa?le? collateraa?le?",
    r"collateral (medial|interne)", r"ligament lateral interne",
]
COLLATERAL_GENERIC = [
    r"collateral ligaments", r"colaterales", r"*seitenbander", r"*kollateralbander", r"kolateraln\w* ligamenti",
    r"πλαγι\w* συνδεσμ\w*", r"колатерални\w* (лигамент|връзк)", r"странични\w* връзки",
    r"kollateral ligamanlar", r"collaterale banden", r"laterale banden", r"ligaments (collateraux|lateraux)",
]
MENISCUS = [
    r"menisc(?!o-?(capsul|femor|tibi))", r"menisk(?!o-?(kapsul|femor))", r"menisq", r"μηνισκ",
    r"мениск", r"*innenmenisk", r"*au(ss|ß)enmenisk", r"*binnenmenisc", r"*buitenmenisc",
]
MENISCUS_PLURAL = [
    r"menisci\b", r"meniscos\b", r"menisken\b", r"meniski\b", r"μηνισκοι", r"μηνισκων", r"менискуси",
    r"менисци", r"meniskusler", r"menisques\b", r"meniscussen",
]
BOTH = [r"both\b", r"ambos\b", r"beide\w*", r"oba\b", r"obje\b", r"αμφοτερ\w*", r"двата\b", r"her iki",
        r"les deux"]

SIDE_MED = [
    r"medial\w*", r"mediaa?l\w*", r"medija\w*", r"intern[oae]s?\b", r"*innen", r"εσω\b", r"медиал\w*", r"binnen\w*",
    r"vnitrn",
]
SIDE_LAT = [
    r"lateral\w*", r"lateraa?l\w*", r"lateraln\w*", r"extern[oae]s?\b", r"*au(ss|ß)en", r"εξω\b", r"латерал\w*",
    r"buiten\w*",
]
PATELLOFEMORAL = [
    r"patel\w*", r"*patell", r"rotul\w*", r"troc?hle\w*", r"trohle\w*", r"*femoropatel", r"*patelofemor",
    r"retropatel\w*", r"επιγονατιδ\w*", r"τροχιλ\w*", r"пател\w*", r"трохле\w*", r"facet\w*", r"faset\w*",
    r"fasett\w*", r"carilla", r"femoro-?patel\w*", r"gleitlager",
]
COMPARTMENT = [r"compart\w*", r"*kompart", r"διαμερισμ\w*", r"компартимент\w*", r"kompartman\w*"]

# ---------- lesiones ----------
TEAR = [
    r"tear\w*", r"torn\b", r"rupt\w*", r"disrupt\w*", r"discontinu\w*", r"avuls\w*", r"lesion\w*",
    r"injur\w*", r"bucket", r"flap\b", r"insufficien\w*", r"deficien\w*",
    r"rotur\w*", r"roto\b", r"rota\b", r"desgarr\w*", r"fisura\w*",
    r"*riss", r"*ruptur", r"lasion\w*", r"insuffizien\w*",
    r"puknu\w*", r"lezij\w*", r"rascjep\w*",
    r"ρηξ\w*", r"ασυνεχ\w*",
    r"руптур\w*", r"разкъс\w*", r"скъс\w*", r"увред\w*", r"лезия\w*",
    r"yirti\w*", r"devamsiz\w*", r"lezyon\w*", r"hasar\w*",
    r"*scheur", r"*letsel", r"*ruptuur", r"laesie\w*",
    r"dechir\w*", r"fissure\b", r"anse de seau", r"desinsertion\w*",
]
SPRAIN = [r"sprain\w*", r"esguince\w*", r"distorsi\w*", r"zerrung\w*", r"istegnu\w*", r"distenzij\w*",
          r"διαστρεμ\w*", r"διαταση\w*", r"разтяг\w*", r"burkul\w*", r"verstuik\w*", r"entorse\w*",
          r"grade (ii|iii|2|3)\b"]
MENISCAL_DEGEN = [
    r"degenera\w*", r"dejenera\w*", r"εκφυλ\w*", r"дегенера\w*", r"mucoid\w*", r"mukoid\w*", r"mucoide\w*",
    r"increased signal", r"aumento de senal", r"*vorschadig", r"signalanhebung", r"grade (i|1)\b",
]

OA_STRONG = [
    r"osteoarthr\w*", r"oa\b", r"arthros\w*", r"artros\w*", r"artroz\w*", r"artrose\w*", r"*gonarthr",
    r"gonartr\w*", r"goanrtr\w*", r"chondropath\w*", r"chondromalac\w*", r"chondrosis", r"condropat\w*",
    r"condromalac\w*", r"hondromalac\w*", r"hondropat\w*", r"kondromala\w*", r"kondropat\w*",
    r"osteophyt\w*", r"osteofit\w*", r"osteofyt\w*", r"osteophyte\w*", r"*osteophyt", r"enthesiophyt\w*",
    r"οστεοαρθρ\w*", r"αρθροπαθ\w*", r"χονδροπαθ\w*", r"χονδρομαλακ\w*", r"οστεοφυτ\w*",
    r"οστεοπαραγωγ\w*", r"гонартроз\w*", r"артроз\w*", r"хондромалац\w*", r"хондропат\w*",
    r"остеофит\w*", r"joint space narrow\w*", r"eklem araligi\w*( \w+)? daral\w*", r"pinzamiento (articular|del espacio|femorotibial)",
    r"ulcera\w* condral\w*", r"defecto\w* condral\w*", r"chondral (defect|loss|lesion|ulcer|injur|thinning|fissur|wear)\w*",
    r"full thickness cartilage", r"cartilage loss",
]
CARTILAGE = [r"cartilag\w*", r"*knorpel", r"hrskavic\w*", r"χονδρ\w*", r"хрущял\w*", r"kikirdak\w*",
             r"kartilaj\w*", r"kraakbeen\w*", r"condral\w*", r"chondral\w*"]
CARTILAGE_DAMAGE = [
    r"loss", r"thin\w*", r"defect\w*", r"fissur\w*", r"erosi\w*", r"denud\w*", r"ulcer\w*", r"wear", r"irregular\w*",
    r"perdida", r"adelgaz\w*", r"desgaste", r"*verlust", r"*defekt", r"*irregularit", r"ausdunn\w*", r"*glatze",
    r"*schaden", r"erozij\w*", r"stanjen\w*", r"ostecen\w*", r"διαβρωσ\w*", r"λεπτυνσ\w*", r"απωλει\w*",
    r"ρωγμ\w*", r"изтъня\w*", r"дефект\w*", r"фисур\w*", r"ерози\w*", r"incel\w*", r"kayb\w*", r"defekt\w*",
    r"verlies\w*", r"verdun\w*", r"amincissement\w*", r"perte\b", r"usure\w*", r"reduc\w* (debljin|thickness|espesor|dicke)\w*", r"(thinned|thinning)", r"dejenera\w*", r"degenera\w*",
]
TRICOMPARTMENTAL = [r"tricompart\w*", r"trikompart\w*", r"τριδιαμερισ\w*"]

EFFUSION = [
    r"effusion\w*", r"hemarthros\w*", r"haemarthros\w*", r"hydrops\b", r"derrame\w*", r"hemartros\w*",
    r"hidrartros\w*", r"*erguss", r"*erguß", r"izljev\w*", r"συλλογη\w*", r"υδραρθρ\w*", r"αιμαρθρ\w*",
    r"излив\w*", r"хидартроз\w*", r"efuzyon\w*", r"(eklem\w*|suprapatellar\w*|intraartikuler\w*|eklemde)( \w+){0,3} sivi artis\w*", r"epanchement\w*", r"hydarthrose\w*",
    r"hemarthrose\w*", r"*hamarthros", r"lipoh?a?emarthros\w*", r"effusie\w*", r"gewrichtsvocht\w*", r"intra-?articulair vocht",
    r"opzetting van (de )?suprapatellaire", r"fluid distension",
]
SYNOVITIS = [
    r"synovit\w*", r"*synovialit", r"sinovit\w*", r"synovial (thicken|prolifer|hypertroph)\w*",
    r"(engrosamiento|proliferacion|hipertrofia) sinovial", r"sinovial\w* (engros|prolifer)\w*",
    r"*synovialisverdick", r"proliferacij\w* sinovij\w*", r"zadebljanj\w* sinovij\w*",
    r"υμενιτιδ\w*", r"παχυνση\w* του υμενα", r"синовит\w*", r"синовиалн\w* пролифер\w*",
    r"sinovyal (kalinlas|prolifer)\w*", r"synovite\w*", r"epaississement synovial",
    r"synoviale (verdikking|proliferatie)", r"synovial membrane thickening", r"membrana sinovial engrosada",
]
BAKER = [
    r"baker\w*", r"popliteal cyst", r"poplit\w* (cyst|cist|kyst|kist|zyst)\w*", r"quiste\w* poplit\w*",
    r"ιγνυακ\w* κυστ\w*", r"подколенн\w* кист\w*", r"kniekuilcyst\w*", r"kyste\w* poplit\w*", r"cist\w* poplit\w*", r"бекер\w*", r"бейкър\w*",
]
CONTUSION = [
    r"contusi\w*", r"bone bruis\w*", r"bruis\w*", r"*kontusion", r"*prellung", r"kontuzi\w*", r"θλασ\w*",
    r"контуз\w*", r"kontuzyon\w*", r"*contusie", r"kneuzing\w*",
]
FRACTURE = [
    r"fractur\w*", r"fraktur\w*", r"*fraktur", r"fractuur\w*", r"*fractuur", r"kirik\w*", r"prijelom\w*",
    r"καταγμ\w*", r"фрактур\w*", r"счупв\w*", r"segond\b", r"knochern\w* ausriss\w*", r"bony avulsion",
    r"avulsion fracture", r"breuk\w*", r"*infraktion",
]

# ---------- modificadores ----------
NEG_PRE = [
    r"no\b", r"not\b", r"without\b", r"negative\b", r"absence\b", r"nor\b", r"neither\b",
    r"free of\b", r"none\b", r"sin\b", r"ni\b", r"ausencia\b", r"kein\w*", r"ohne\b", r"nicht\b",
    r"bez\b", r"nema\b", r"nije\b", r"ne\b", r"niti\b", r"δεν\b", r"χωρις\b", r"ανευ\b", r"ουτε\b",
    r"μη\b", r"няма\b", r"без\b", r"не\b", r"geen\b", r"niet\b", r"zonder\b", r"noch\b", r"pas\b",
    r"sans\b", r"aucun\w*", r"neg\b",
]
NEG_POST = [
    r"yok\w*", r"izlenme\w*", r"saptanma\w*", r"gorulme\w*", r"rastlanma\w*", r"degil\w*",
    r"not seen", r"not identified", r"not demonstrated", r"excluded", r"ausgeschlossen",
    r"nicht nachweisbar", r"descartad\w*", r"none\b", r"negative\b", r"absent\b", r"not present",
    r"nil\b", r"keine?\b", r"ninguno\w*", r"nema\b", r"няма\b", r"yok\b", r"geen\b", r"aucun\w*",
]
DISPLACEMENT = [r"displac\w*", r"verplaat\w*", r"disloz\w*", r"desplaz\w*", r"dislok\w*", r"дислок\w*",
                r"μετατοπ\w*", r"deplac\w*", r"pomak\w*", r"significa\w*", r"significant\w*", r"значим\w*"]
UNCERTAIN = [
    r"suspect\w*", r"suspicio\w*", r"possibl\w*", r"probabl\w*", r"likely\b", r"may\b", r"might\b",
    r"question\w*", r"equivocal", r"cannot be excluded", r"can'?t (be |totally )?exclud\w*", r"r/?o\b",
    r"rule out", r"differential", r"versus\b", r"vs\b", r"posibl\w*", r"sospech\w*", r"dudos\w*",
    r"no se descarta", r"a descartar", r"fraglich\w*", r"moglich\w*", r"wahrscheinlich\w*", r"verdacht\w*",
    r"vermutlich", r"v\. ?a\b", r"wohl\b", r"dd\b", r"nicht sicher", r"eventuell", r"evtl\b", r"moguc\w*", r"vjerojatn\w*",
    r"suspekt\w*", r"sumnj\w*", r"upitn\w*", r"dif\b", r"πιθαν\w*", r"υποψι\w*", r"ενδεχομ\w*", r"μπορουσε",
    r"вероятн\w*", r"възможн\w*", r"съмнени\w*", r"olasi\w*", r"muhtemel\w*", r"suphe\w*", r"ekarte edilemez",
    r"mogelijk\w*", r"waarschijnlijk\w*", r"verdacht\w*", r"twijfel\w*", r"vermoedelijk\w*", r"eventueel",
    r"pourrait", r"evoqu\w*", r"douteu\w*", r"*\?",
]
NORMAL = [
    r"normal\w*", r"intact\w*", r"unremarkable", r"preserved", r"maintained", r"within normal limits",
    r"continuous", r"conservad\w*", r"preservad\w*", r"integr[oa]s?\b", r"continu[oa]s?\b",
    r"sin alteracion\w*", r"sin hallazgos", r"indemne\w*", r"intakt\w*", r"regelrecht\w*", r"unauffallig\w*",
    r"erhalten\w*", r"ohne befund", r"uredn\w*", r"uredan\b", r"odrzan\w*", r"ocuvan\w*", r"primjeren\w*",
    r"neupadljiv\w*", r"bez osobitosti", r"φυσιολογικ\w*", r"ακεραι\w*", r"κανονικ\w*",
    r"δεν παρατηρουνται αξιολογα", r"нормал\w*", r"запазен\w*", r"съхранен\w*", r"правилн\w*",
    r"без особености", r"б о\b", r"dogal\w*", r"korunmus\w*", r"saglam\w*", r"olagan\w*", r"normaal\w*",
    r"zonder bijzonderheden", r"gaaf\b", r"sans particularite", r"respect\w*", r"preserve\w*",
]
INT_MIN = [r"minim\w*", r"trace\b", r"physiolog\w*", r"geringfugig\w*", r"минимал\w*", r"ελαχιστ\w*",
           r"eser\b", r"spoor\w*", r"fisiologic\w*", r"superficial\w*", r"oberflachlich\w*",
           r"povrsn\w*", r"grade (i|1)\b", r"grado (i|1)\b", r"(i|1)\.? stupnj\w*", r"low[- ]grade", r"laaggradig\w*", r"stade (i|1)\b"]
INT_MILD = [r"mild\w*", r"small\b", r"slight\w*", r"little\b", r"leve\b", r"discret\w*", r"peque\w*",
            r"escas\w*", r"gering\w*", r"leicht\w*", r"klein\w*", r"blag\w*", r"manj\w*", r"mal[ia]\b",
            r"ηπι\w*", r"μικρ\w*", r"лек\w*", r"малк\w*", r"малък", r"незначител\w*", r"hafif\w*", r"az\b",
            r"leger\w*", r"petit\w*", r"licht\w*", r"beperkt\w*", r"some amount"]
INT_MOD = [r"moderat\w*", r"large\b", r"massive\b", r"marked\w*", r"extensive\w*", r"significant\w*",
           r"moderad\w*", r"abundante\w*", r"important\w*", r"masiv\w*", r"maßig\w*", r"massig\w*",
           r"deutlich\w*", r"ausgepragt\w*", r"gro(ss|ß)\w*", r"erheblich\w*", r"umjeren\w*", r"opsez\w*",
           r"izrazit\w*", r"znacaj\w*", r"μετρι\w*", r"μεγαλ\w*", r"σημαντικ\w*", r"ευμεγεθ\w*", r"умерен\w*",
           r"голям\w*", r"значител\w*", r"изразен\w*", r"orta\b", r"belirgin\w*", r"fazla\w*", r"artmis\w*",
           r"yogun\w*", r"modere\w*", r"abondant\w*", r"matig\w*", r"uitgebreid\w*", r"groot\w*",
           r"aanzienlijk\w*", r"severe\w*", r"advanced", r"high[- ]grade", r"full[- ]thickness", r"severo\w*",
           r"grade (ii|iii|iv|2|3|4)\b", r"grado (ii|iii|iv|2|3|4)\b", r"iii°", r"uznapredoval\w*"]

# Encabezados de sección. Los de indicación clínica se descartan (no describen la imagen).
HEADERS_DROP = [
    r"antecedentes clinicos", r"indicacion\w*", r"indication\w*", r"clinical (history|information|indication)\w*",
    r"history\b", r"klinische inlichtingen", r"diagnostische vraagstel\w*", r"fragestellung", r"klinische angaben",
    r"indikation", r"uputn\w* dijagnoz\w*", r"klinicki podaci", r"endikasyon", r"klinik bilgi", r"показани\w*",
    r"renseignements cliniques", r"κλινικ\w* πληροφορ\w*", r"ενδειξ\w*", r"aa\b",
]
HEADERS_KEEP = [
    r"findings", r"impression", r"conclusion", r"hallazgos", r"impresion", r"conclusion\w*", r"befund\w*",
    r"beurteilung", r"nalaz", r"zakljucak", r"ευρηματα", r"συμπερασμα", r"заключение", r"мр находка",
    r"bulgular", r"sonuc", r"constatations", r"bevindingen", r"besluit", r"resultados", r"technique",
    r"technik", r"tecnica", r"τεχνικη",
]
