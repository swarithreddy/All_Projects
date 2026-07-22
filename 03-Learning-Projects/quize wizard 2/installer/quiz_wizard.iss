; Inno Setup script for Quiz Wizard 2.0.0
; Compile after: python -m PyInstaller pyinstaller.spec --noconfirm

#define MyAppName "Quiz Wizard"
#define MyAppVersion "2.0.0"
#define MyAppPublisher "Quiz Wizard"
#define MyAppExeName "QuizWizard.exe"

[Setup]
AppId={{8F3C2A71-9B4E-4D6A-A1C2-QuizWizard200}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={localappdata}\Programs\QuizWizard
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=..\dist\installer
OutputBaseFilename=QuizWizard-Setup-2.0.0
SetupIconFile=..\assets\app.ico
Compression=lzma
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\{#MyAppExeName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional icons:"; Flags: unchecked

[Files]
; Application binaries (do not touch user AppData leaderboard)
Source: "..\dist\QuizWizard\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent

[Code]
function InitializeUninstall(): Boolean;
begin
  Result := True;
  MsgBox('Quiz Wizard will be removed from Programs.'#13#10 +
         'Your scores in %LOCALAPPDATA%\QuizWizard are kept by default.',
         mbInformation, MB_OK);
end;
