add_executable(ppcdis ppcdis/main.cpp)
target_link_libraries(ppcdis PRIVATE wiiu::cpu)
wiiu_set_warnings(ppcdis)
set_target_properties(ppcdis PROPERTIES FOLDER "tools")

add_executable(rpl-dump rpldump/main.cpp)
target_link_libraries(rpl-dump PRIVATE wiiu::loader wiiu::cpu)
wiiu_set_warnings(rpl-dump)
set_target_properties(rpl-dump PROPERTIES FOLDER "tools")
