# Error Analysis Report: TF-IDF Model vs. Human Judgment

## Case 22
**Why the Human was right:** Both the Anchor and Story B focus on a female protagonist suffering from a debilitating neurological/psychological condition that distorts their perception of reality. This vulnerability is central to a sinister plot by external forces. The core Abstract Theme is the exploitation of a compromised mind. Story A is a straightforward "mistaken identity" crime story, not a psychological thriller based on exploiting the victim's perception.

**Why the TF-IDF Model failed:** The model was fooled by explicit keywords "amnesia" and "hospital" in Story A, which are direct lexical hits. It latched onto the concrete medical condition but missed the abstract narrative function of psychological torment and manipulation, which was stronger in Story B.

**Error Category:** Abstract Concept Blindness

## Case 124
**Why the Human was right:** The Anchor and Story B share a specific Abstract Theme: a protagonist managing a low-level, morally ambiguous drug-dealing side job while trying to maintain a normal life. The central conflict comes from the instability of this criminal hustle. Story A is a more extreme, violent crime narrative lacking the "stuck between two worlds" theme.

**Why the TF-IDF Model failed:** The model was drawn to significant keyword overlap in Story A related to youth angst, family problems (`father`), and relationships (`love`, `girl`). These common, high-frequency words overshadowed the more specific but narratively crucial keywords like `drug` and `medication` that linked the Anchor to Story B.

**Error Category:** Keyword Trap (Setting)

## Case 46
**Why the Human was right:** The Anchor and Story B share an identical core Course of Action: (1) A spouse discovers an affair, (2) in a moment of passion, they kill someone connected to the affair, and (3) the story deals with the protagonist facing consequences. Story A focuses on sexual obsession and mutilation, not the legal/personal aftermath for the killer.

**Why the TF-IDF Model failed:** The model was distracted by unique, non-overlapping vocabulary in Story B related to the escape plot (`identity`, `Yugoslavia`, `Adriatic coast`), which lowered cosine similarity. Story A had a higher concentration of shared keywords like `affair`, `killed`, making it appear more similar despite its different narrative arc.

**Error Category:** Abstract Concept Blindness

## Case 164
**Why the Human was right:** Both the Anchor and Story B are domestic dramas centered on family and romantic relationships that lead to tragic, destructive outcomes. The core Abstract Theme is personal relationships culminating in ruin and death/separation. Story A's tragedy is detached political satire where main characters are unharmed.

**Why the TF-IDF Model failed:** The model was deceived by strong keyword overlap related to professional settings. The Anchor is about a "writer" and "memory play," while Story A involves a "celebrated playwright" and "writer-poet." This shared arts vocabulary created a superficial connection.

**Error Category:** Keyword Trap (Setting)

## Case 25
**Why the Human was right:** The definitive link is the highly unusual Abstract Theme of characters who are, or believe themselves to be, animals. The Anchor features anthropomorphic animals; Story A is about humans with zoanthropy believing they're animals. A secondary shared theme is institutional confinement and escape.

**Why the TF-IDF Model failed:** The model fell into a keyword trap based on sexuality and classical settings. The Anchor contains `Marquis de Sade`, `penis`, `rape`; Story B contains `amorous pleasures`, `god Priapus`, set in `empire of Nero`. This debauchery vocabulary created a stronger signal than the abstract "animal identity" theme.

**Error Category:** Abstract Concept Blindness

## Case 97
**Why the Human was right:** Both the Anchor and Story B share the Abstract Theme of a protagonist confronting devastating consequences of a close family member's past. They're dramas about the heavy, inescapable weight of family history. Story A is a standard coming-of-age tale about a young man's future.

**Why the TF-IDF Model failed:** The model was misled by nearly identical socio-economic setting keywords: `middle-class`, `wealthy`, `suburb`, `parents`, `son`, `college`, `wife`, `girlfriend`, `country-club`. It identified the lifestyle perfectly but missed the narrative substance.

**Error Category:** Keyword Trap (Setting)

## Case 79
**Why the Human was right:** The core connection is the Course of Action: a protagonist develops an obsessive, psychologically complex relationship with the person who committed a crime against their family member. Both narratives focus on the tormented psychology of the survivor grappling with the perpetrator. Story A is straightforward horror about possession.

**Why the TF-IDF Model failed:** Keyword overlap between Anchor and Story A was overwhelming: `daughter`, `mother`, `father`, `rape`/`murder`, plus psychological trauma terms (`insanity`, `psychotic trance`, `psychiatrist`). The model picked the most direct keyword parallels rather than the most similar narrative structure.

**Error Category:** Keyword Trap (Setting)

## Case 18
**Why the Human was right:** The Anchor and Story A share the Abstract Theme of a young, hopeful protagonist being destroyed by a cruel, powerful, unjust force, leading to tragedy. Story B shares class-based romantic struggle but has a happy Outcome, making it fundamentally different.

**Why the TF-IDF Model failed:** The model was unable to distinguish between setup and outcome. It saw strong keyword overlap in the setup (young woman, class difference, father opposition, running away) and ignored the completely opposite endings (suicide vs. living happily).

**Error Category:** Outcome-Negation Blindness

## Case 89
**Why the Human was right:** The Anchor and Story A share the Abstract Theme of a prominent historical figure fighting against imprisonment/oppression by a foreign power. Both narratives are about political confinement and struggle for freedom. Story B lacks the crucial political "imprisonment" element.

**Why the TF-IDF Model failed:** The model was caught in a keyword trap related to geography. The Anchor is set in "Korea" under "Japanese occupation." Story B features "Japanese seamen." The word "Japanese" created a powerful lexical link that caused it to ignore the actual narrative.

**Error Category:** Keyword Trap (Setting)

## Case 198
**Why the Human was right:** Both the Anchor and Story B are biopics sharing the Abstract Theme of exploring the life, work, and legacy of an influential Eastern philosopher/teacher. Story A is contemporary personal discovery, not a historical biography.

**Why the TF-IDF Model failed:** The model was misled by specific cultural keywords. The Anchor mentions "yoga," "hinduistic beliefs," "temple." Story A contains "Koovagam Festival" re-enacting "Mahabharata." The direct connection between "Hinduistic" and "Mahabharata" was a stronger signal than the abstract "biopic of a teacher" theme.

**Error Category:** Keyword Trap (Setting)

## Case 144
**Why the Human was right:** The Anchor and Story A share the Abstract Theme of romantic lives of working/middle-class people in realistic French settings. They're slice-of-life domestic dramas about difficulty of love after past hurts. Story B is a more dramatic romance with different tone and scope.

**Why the TF-IDF Model failed:** The model got confused by family structure keywords. The Anchor features protagonist with "daughter" and "son." Story B features a "single father" with "young girl" meeting another "single parent." Overlap of `daughter`, `father`, `parent` created higher similarity for Story B.

**Error Category:** Keyword Trap (Setting)

## Case 81
**Why the Human was right:** The Anchor and Story A share the Abstract Theme of an aging artist grappling with career, legacy, and relevance. Both deal with challenges of being a creator past one's prime. Story B is about personal/cultural identity, not the life of an artist.

**Why the TF-IDF Model failed:** The model was led astray by themes of memory and family history. The Anchor mentions protagonist "profoundly affected by death of his father." Story B is about journey to "scatter parents' ashes" and connect with "memories." Strong lexical overlap of `father`/`parents`, `death`/`ashes`, `memories` created an incorrect match.

**Error Category:** Abstract Concept Blindness

## Case 172
**Why the Human was right:** The central narrative element is the Course of Action: protagonist in established marriage has illicit affair and ultimately decides to leave current partner for new person. In Story B, protagonist kills impulsively then escapes; he's not a long-term plotter like in Anchor and Story A.

**Why the TF-IDF Model failed:** The model was heavily biased by crime vocabulary. Anchor contains `gangsters`, `rob`, `police`, `revenge`, `shoot`. Story B is filled with `murdered`, `police`, `menacing people`. This "crime thriller" setting overlap created high similarity score, while Story A's romantic drama had almost no crime vocabulary.

**Error Category:** Keyword Trap (Setting)

## Case 190
**Why the Human was right:** The Anchor and Story A share the Abstract Theme of an outsider entering a domestic environment and forming a class-crossing romantic relationship that leads to social integration. Both are stories about finding a place within a family structure through a relationship that bridges a social gap. Story B is a survival story about people removed from society.

**Why the TF-IDF Model failed:** The model was fooled by surface-level setting keywords. The Anchor begins with man found "out cold on a beach." Story B is about people "lost at sea in a rowboat." The lexical connection between "beach" and "sea" created very high similarity (0.097 vs 0.000 for A), missing the core social narrative.

**Error Category:** Keyword Trap (Setting)

## Case 143
**Why the Human was right:** The core connection is the Abstract Theme focusing on female relationships and community. The Anchor's plot is driven by dynamic between two new female friends. Story A is a documentary explicitly about a community of women. Both are fundamentally about female social dynamics. Story B is a crime thriller about a mystery, not about relationships between women.

**Why the TF-IDF Model failed:** The model was deceived by shared location keywords. The Anchor is set in "Paris" in a "new apartment." Story B is also set in "Paris" in an "apartment." This repeated overlap created a misleading signal. Story A had no geographic keywords linking it to the anchor.

**Error Category:** Keyword Trap (Setting)

## Case 169
**Why the Human was right:** The Anchor and Story A share a specific Course of Action: protagonist devises and executes a deceptive, high-stakes plot against a wealthy individual for personal gain. In both, protagonist is the active plotter who insinuates into victim's life. Story B's protagonist reacts impulsively, then escapes; he's not a long-term plotter.

**Why the TF-IDF Model failed:** The model's prediction was extremely close (0.023 vs 0.024). It couldn't grasp the abstract role of the protagonist ("the plotter"). It matched on general "crime and betrayal" keywords: Anchor has "imprisoned for a crime," "rich lover," "revenge"; Story B has "affair," "lover," "fatal mistake." Combined crime/betrayal keywords in B were slightly stronger than in A.

**Error Category:** Abstract Concept Blindness

## Case 61
**Why the Human was right:** The Anchor and Story B share a specific Course of Action: a group of greedy men seeking "gold," with a central female character as desired prize, facing threat from external group ("natives" / "bandits"). Plot is driven by twin desires for treasure and control over woman. Story A is a Western revenge plot where wife's kidnapping is the goal itself.

**Why the TF-IDF Model failed:** Prediction was razor-thin (0.112 vs 0.110). The model fell into a genre keyword trap. The Tarzan story has adventure-pulp feel. Story A is saturated with generic Western keywords (`gunrunner`, `American Civil War`, `outlaws`, `sheriff`, `kidnapping`). This density made it appear slightly more similar than Story B's specific "gold mine" and "bandits" elements.

**Error Category:** Keyword Trap (Setting)

## Case 165
**Why the Human was right:** Both Anchor and Story B are murder mysteries set in "Paris" where crime originates from dark, illicit sexual relationship. They share the Abstract Theme of Parisian noir, where sex and death are intertwined. Story A is psychological horror about sadomasochism, not a "whodunit."

**Why the TF-IDF Model failed:** The model fixated on a very specific plot detail: voyeurism. Anchor states protagonist was recruited "to watch a much older man's sexual liaison." Story A is filled with `gigolo`, `prostitutes`, `records everything`, `likes to watch`. This highly specific concept created a powerful lexical link that outweighed the general "murder mystery in Paris" theme.

**Error Category:** Keyword Trap (Setting)

## Case 98
**Why the Human was right:** The Anchor and Story A share a core Course of Action: one spouse's affair prompts the other to seek their own affair in response. This reciprocal reaction is the central narrative driver. Story B revolves around sacrifice and reconciliation with terminal illness, a completely different structure.

**Why the TF-IDF Model failed:** The model was overwhelmed by heavy dramatic vocabulary in Story B: `dying`, `debilitating disease`, `epidemic`, `typhoid`, `quarantined`, `accepts death`. This appeared more similar to Anchor's emotional turmoil than Story A's lighter comedic tone of `flirtation`. It matched emotional weight, not plot mechanics.

**Error Category:** Abstract Concept Blindness

## Case 75
**Why the Human was right:** Both Anchor and Story A feature a protagonist who is an obsessive outsider to another couple's infidelity, whose meddling leads to tragic outcome. The Abstract Theme is destructive consequences of vicarious involvement. Story B's protagonist is a direct participant (the killer), not an observer.

**Why the TF-IDF Model failed:** Catastrophic failure due to keyword trap. The protagonist in Anchor is named **"Paul."** The protagonist in Story B is also named  **"Paul."**  This identical proper noun, combined with `wife` and `affair`, created overwhelmingly high similarity (0.328 vs 0.077 for A). The model found the most superficial connection and ignored narrative structure.

**Error Category:** Keyword Trap (Setting)

## Case 14
**Why the Human was right:** The human recognizes shared satirical/comedic tone. Both Anchor and Story B follow the Abstract Theme of an incompetent but ultimately successful protagonist ("failing upwards"). Story A, a serious historical drama about Napoleon's imprisonment, is the thematic opposite.

**Why the TF-IDF Model failed:** The model was trapped by dense vocabulary of politics and power. Anchor contains `president`, `United States`, `Vice President`, `Ambassador`, `General`. Story A contains `Napoleon`, `Emperor`, `imprisoned`, `military authorities`. This overlap in "political/military leader" domain made Story A seem perfect, causing the model to miss the crucial comedic tone.

**Error Category:** Keyword Trap (Setting)

## Case 45
**Why the Human was right:** The Anchor and Story B share a specific Course of Action: complex undercover plot within Parisian criminal underworld where things are not as they seem. Both are stories of elaborate deception and infiltration. Story A is a straightforward hostage-and-shootout plot lacking infiltration element.

**Why the TF-IDF Model failed:** The model was drawn to vocabulary of direct confrontation. Anchor mentions "FBI." Story A is filled with `kidnap`, `police officer`, `undercover agent`, `Commissioner`, `attack`, `shootout`. This created stronger "cops and robbers" signal than the subtle "underworld scheme" language in Story B.

**Error Category:** Keyword Trap (Setting)

## Case 170
**Why the Human was right:** The Anchor and Story A are both centered on the Abstract Theme of a young boy's earnest quest driven by innocent motivation. The narratives are defined by child protagonists' pure intentions. Story B is a complex adult melodrama of romance and revenge.

**Why the TF-IDF Model failed:** The model was unable to see the "child's perspective" theme and instead matched on adult concepts. Anchor contains `aristocrat`, `love`, `gang`, `fight`. Story B contains `Countess`, `Baron`, `love`, `criminal`. This overlap in nobility, romance, and crime vocabulary created stronger connection than the thematically-linked story in A.

**Error Category:** Abstract Concept Blindness

## Case 30
**Why the Human was right:** The human identifies the shared Abstract Theme of transgressive and manipulative seduction that disrupts established social order. Both narratives are driven by power dynamics of seduction. Story A is a murder mystery about community justice, fundamentally different.

**Why the TF-IDF Model failed:** The model got confused by concrete "crime" vocabulary. Anchor's theme of "revenge" could be lexically associated with severe crimes like "murder." Presence of `tyrannical`, `murdered`, `investigating judge` in Story A created a "crime of passion" signal that model latched onto, failing to understand nuanced psychological manipulation connecting Anchor to Story B.

**Error Category:** Abstract Concept Blindness

## Case 106
**Why the Human was right:** The Anchor and Story A are thematically identical as allegories about disillusionment with failing/corrupt Soviet-era communist system. The core Abstract Theme is the individual crushed by dysfunctional state bureaucracy. Story B is a personal coming-of-age story, not a critique of the system.

**Why the TF-IDF Model failed:** Catastrophic failure with both similarity scores at zero. The model had no way to connect abstract concepts because specific vocabulary was entirely different. TF-IDF only sees different words and found no shared keywords, completely unable to make meaningful choice. Perfect demonstration of inability to grasp abstract political/social critiques.

**Error Category:** Abstract Concept Blindness