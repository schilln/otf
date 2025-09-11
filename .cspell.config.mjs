/*
 * Requires CSpell Version 8.0.0 or later
 * VSCode Spell Checker Version 4.0.0 or later
 */
import fs from "node:fs";
import path from "node:path";

const _dictDir = ".cspell";

async function getDictionaries() {
  const filenames = (await fs.promises.readdir(_dictDir))
    .filter((f) => f.toLowerCase().endsWith(".txt"))
    .sort();

  return filenames;
}

async function createConfig() {
  const dictFilenames = await getDictionaries();

  const dictionaryDefinitions = dictFilenames.map((dict) => ({
    name: path.parse(dict).name,
    path: path.join(_dictDir, dict),
  }));
  const dictionaries = dictionaryDefinitions.map((dict) => dict.name);

  return {
    dictionaryDefinitions,
    dictionaries,
  };
}

export default createConfig();
