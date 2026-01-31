def parse_model_filename(filename):
    # rimuove estensione .pt se presente
    name = filename.replace(".pt", "")

    parts = name.split("_")
    values = {}

    for part in parts:
        if part.startswith("patchconv"):
            values["patch_conv1d"] = part.replace("patchconv", "") == "True"
        elif part.startswith("spacing"):
            values["spacing"] = int(part.replace("spacing", ""))
        elif part.startswith("spectoks"):
            values["num_special_tokens"] = int(part.replace("spectoks", ""))
        elif part.startswith("vocabsize"):
            values["vocab_size"] = int(part.replace("vocabsize", ""))

    return values
