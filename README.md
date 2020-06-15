# Arduino-ESP-BIP39

Arduino + ESP8266/ESP32 Bip39 compatible mnemonic word generator for (hashed) user supplied entropy of ANY kind
Reason for creating this test file: *I* can't get bip39 conform results using other implementations (trezor/ubitcoin)

Extract: use sha256 on your passphrase, password or binary file to create a BIP39 compatible word list

Advantage: remember your passphrase, password or binary file instead of the actual mnemonic wordlist
Disadvantage: allows you to remember an easy to brute force and thus unsafe password/passphrase etc.

To use any user supplied entropy and be safe you will have to use your wits. Generate the data from dice rolls etc. and you are safe,
but you won't be able to remember what you rolled.
Generate the data from names, common street slang, any previously published remark etc. and you are unsafe.
>Google your passphrase and you are unsafe.<
If you do want to use a passphrase you best use at least one word that is not in any dictionary, e.g.
The quick brown fox jumps over the lazy dog -> Your bitcoin is gone faster than you can type this in.
The quick brown foxy-girl jumps over the lazy dog -> you are probably safe
The slow moving heffalumpski jumps over the lazy dog -> good luck to all brute forcers out there!
It would be even better to not have your passphrase have any likeness to things said/used before. 
