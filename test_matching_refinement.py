
import re

def get_keywords(text):
    stopwords = {"set", "ensure", "minimum", "maximum", "enable", "disable", "is", "to", "and", "the", "parameters", "machine", "software", "system", "currentcontrolset", "services"}
    keywords = set(w.lower() for w in re.findall(r'[a-zA-Z][a-z]*|[A-Z]+(?=[A-Z][a-z]|$)', text))
    return {w for w in keywords if len(w) > 2 and w not in stopwords}

def test_matching_logic(key_name, rule_titles):
    print(f"\nTesting Key: {key_name}")
    
    path_parts = [p.strip() for p in key_name.replace("/", "\\").split("\\") if p.strip()]
    if not path_parts: path_parts = [key_name]
    
    leaf_node = path_parts[-1]
    context_tokens = path_parts[:-1][::-1]
    leaf_keywords = get_keywords(leaf_node)
    
    print(f"Leaf: {leaf_node} | Keywords: {leaf_keywords}")
    
    candidates = []
    for title in rule_titles:
        title_lower = title.lower()
        
        # Strict Match: Must contain at least one leaf keyword
        if not any(kw in title_lower for kw in leaf_keywords):
            continue
            
        # Context Score
        score = 0
        for i, token in enumerate(context_tokens):
            if token.lower() in title_lower:
                score += (10 / (i + 1))
        
        candidates.append({"title": title, "score": score})
    
    candidates.sort(key=lambda x: x["score"], reverse=True)
    
    if not candidates:
        print("Result: SKIP (No related rules found)")
    else:
        print("Candidates (sorted by score):")
        for c in candidates:
            print(f" - [{c['score']:.1f}] {c['title']}")
        print(f"Best Match: {candidates[0]['title']}")

if __name__ == "__main__":
    titles = [
        "Ensure 'Digitally sign client communication' is set to 'Enabled'",
        "Ensure 'Netlogon: Digitally sign secure channel communication' is set to 'Enabled'",
        "Ensure 'Minimum password age' is set to '1 or more days'",
        "Ensure 'Network access: Restrict anonymous access' is set to 'Enabled'"
    ]
    
    test_matching_logic("MACHINE\\System\\CurrentControlSet\\Services\\Netlogon\\Parameters\\SignSecureChannel", titles)
    test_matching_logic("MACHINE\\Software\\Microsoft\\Windows\\CurrentVersion\\Policies\\System\\MinimumPasswordAge", titles)
    test_matching_logic("MACHINE\\System\\CurrentControlSet\\Control\\Lsa\\RestrictAnonymous", titles)
    test_matching_logic("MACHINE\\Unrelated\\Path\\RandomKey", titles)
